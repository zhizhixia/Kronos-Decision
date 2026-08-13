import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
import sys
import hashlib
from typing import Any

from tqdm import trange

sys.path.append("../")
from model.module import *
from model.prediction import PredictionPaths, ensure_before_deadline, sampling_params_hash


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    KronosTokenizer module for tokenizing input data using a hybrid quantization approach.

    This tokenizer utilizes a combination of encoder and decoder Transformer blocks
    along with the Binary Spherical Quantization (BSQuantizer) to compress and decompress input data.

    Args:
           d_in (int): Input dimension.
           d_model (int): Model dimension.
           n_heads (int): Number of attention heads.
           ff_dim (int): Feed-forward dimension.
           n_enc_layers (int): Number of encoder layers.
           n_dec_layers (int): Number of decoder layers.
           ffn_dropout_p (float): Dropout probability for feed-forward networks.
           attn_dropout_p (float): Dropout probability for attention mechanisms.
           resid_dropout_p (float): Dropout probability for residual connections.
           s1_bits (int): Number of bits for the pre token in BSQuantizer.
           s2_bits (int): Number of bits for the post token in BSQuantizer.
           beta (float): Beta parameter for BSQuantizer.
           gamma0 (float): Gamma0 parameter for BSQuantizer.
           gamma (float): Gamma parameter for BSQuantizer.
           zeta (float): Zeta parameter for BSQuantizer.
           group_size (int): Group size parameter for BSQuantizer.

    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):

        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits # Total dimension of the codebook after quantization
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder Transformer Blocks
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        # Decoder Transformer Blocks
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim) # Linear layer before quantization
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model) # Linear layer after quantization (pre part - s1 bits)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model) # Linear layer after quantization (full codebook)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size) # BSQuantizer module

    def forward(self, x):
        """
        Forward pass of the KronosTokenizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            tuple: A tuple containing:
                - tuple: (z_pre, z) - Reconstructed outputs from decoder with s1_bits and full codebook respectively,
                         both of shape (batch_size, seq_len, d_in).
                - torch.Tensor: bsq_loss - Loss from the BSQuantizer.
                - torch.Tensor: quantized - Quantized representation from BSQuantizer.
                - torch.Tensor: z_indices - Indices from the BSQuantizer.
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z) # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits] # Extract the first part of quantized representation (s1_bits)
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        # Decoder layers (for pre part - s1 bits)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Decoder layers (for full codebook)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """
        Converts indices to bit representations and scales them.

        Args:
            x (torch.Tensor): Indices tensor.
            half (bool, optional): Whether to process only half of the codebook dimension. Defaults to False.

        Returns:
            torch.Tensor: Bit representation tensor.
        """
        if half:
            x1 = x[0] # Assuming x is a tuple of indices if half is True
            x2 = x[1]
            mask = 2 ** torch.arange(self.codebook_dim//2, device=x1.device, dtype=torch.long) # Create a mask for bit extraction
            x1 = (x1.unsqueeze(-1) & mask) != 0 # Extract bits for the first half
            x2 = (x2.unsqueeze(-1) & mask) != 0 # Extract bits for the second half
            x = torch.cat([x1, x2], dim=-1) # Concatenate the bit representations
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long) # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0 # Extract bits

        x = x.float() * 2 - 1 # Convert boolean to bipolar (-1, 1)
        q_scale = 1. / (self.codebook_dim ** 0.5) # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        Encodes the input data into quantized indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).
            half (bool, optional): Whether to use half quantization in BSQuantizer. Defaults to False.

        Returns:
            torch.Tensor: Quantized indices from BSQuantizer.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        """
        Decodes quantized indices back to the input data space.

        Args:
            x (torch.Tensor): Quantized indices tensor.
            half (bool, optional): Whether the indices were generated with half quantization. Defaults to False.

        Returns:
            torch.Tensor: Reconstructed output tensor of shape (batch_size, seq_len, d_in).
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Kronos(nn.Module, PyTorchModelHubMixin):
    """
    Kronos Model.

    Args:
        s1_bits (int): Number of bits for pre tokens.
        s2_bits (int): Number of bits for post tokens.
        n_layers (int): Number of Transformer blocks.
        d_model (int): Dimension of the model's embeddings and hidden states.
        n_heads (int): Number of attention heads in the MultiheadAttention layers.
        ff_dim (int): Dimension of the feedforward network in the Transformer blocks.
        ffn_dropout_p (float): Dropout probability for the feedforward network.
        attn_dropout_p (float): Dropout probability for the attention layers.
        resid_dropout_p (float): Dropout probability for residual connections.
        token_dropout_p (float): Dropout probability for token embeddings.
        learn_te (bool): Whether to use learnable temporal embeddings.
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing for s1 decoding. Defaults to False.
            s1_targets (torch.Tensor, optional): Target s1 token IDs for teacher forcing. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - s2_logits: Logits for s2 token predictions, conditioned on s1. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask) # Dependency Aware Layer: Condition on s1 embeddings
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Decodes only the s1 tokens.

        This method performs a forward pass to predict only s1 tokens. It returns the s1 logits
        and the context representation from the Transformer, which can be used for subsequent s2 decoding.

        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - context: Context representation from the Transformer. Shape: [batch_size, seq_len, d_model]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """
        Decodes the s2 tokens, conditioned on the context and s1 tokens.

        This method decodes s2 tokens based on a pre-computed context representation (typically from `decode_s1`)
        and the s1 token IDs. It uses the dependency-aware layer and the conditional s2 head to predict s2 tokens.

        Args:
            context (torch.Tensor): Context representation from the transformer (output of decode_s1).
                                     Shape: [batch_size, seq_len, d_model]
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            torch.Tensor: s2 logits. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True, generator=None):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1, generator=generator)

    return x


def _prepare_autoregressive_state(
    tokenizer: Any, x: torch.Tensor, x_stamp: torch.Tensor,
    y_stamp: torch.Tensor, max_context: int, pred_len: int,
    sample_count: int,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, int, int]:
    device = x.device
    x = x.unsqueeze(1).repeat(1, sample_count, 1, 1)
    x = x.reshape(-1, x.size(2), x.size(3)).to(device)
    x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1)
    x_stamp = x_stamp.reshape(-1, x_stamp.size(2), x_stamp.size(3)).to(device)
    y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1)
    y_stamp = y_stamp.reshape(-1, y_stamp.size(2), y_stamp.size(3)).to(device)

    x_token = tokenizer.encode(x, half=True)
    initial_seq_len = x.size(1)
    total_seq_len = initial_seq_len + pred_len
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
    batch_size = x_token[0].size(0)
    generated_pre = x_token[0].new_empty(batch_size, pred_len)
    generated_post = x_token[1].new_empty(batch_size, pred_len)
    pre_buffer = x_token[0].new_zeros(batch_size, max_context)
    post_buffer = x_token[1].new_zeros(batch_size, max_context)
    buffer_len = min(initial_seq_len, max_context)
    if buffer_len > 0:
        start_idx = max(0, initial_seq_len - max_context)
        pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
        post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]
    return (
        x_token, full_stamp, generated_pre, generated_post,
        pre_buffer, post_buffer, initial_seq_len, total_seq_len,
    )


def _autoregressive_context(
    pre_buffer: torch.Tensor, post_buffer: torch.Tensor,
    full_stamp: torch.Tensor, current_seq_len: int, max_context: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    window_len = min(current_seq_len, max_context)
    if current_seq_len <= max_context:
        input_tokens = [
            pre_buffer[:, :window_len],
            post_buffer[:, :window_len],
        ]
    else:
        input_tokens = [pre_buffer, post_buffer]
    context_start = max(0, current_seq_len - max_context)
    current_stamp = full_stamp[:, context_start:current_seq_len, :].contiguous()
    return input_tokens, current_stamp


def _sample_autoregressive_pair(
    model: Any, input_tokens: list[torch.Tensor], current_stamp: torch.Tensor,
    temperature: float, top_k: int, top_p: float,
    generator: torch.Generator | None, deadline: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    s1_logits, context = model.decode_s1(
        input_tokens[0], input_tokens[1], current_stamp,
    )
    ensure_before_deadline(deadline)
    sample_pre = sample_from_logits(
        s1_logits[:, -1, :], temperature=temperature, top_k=top_k,
        top_p=top_p, sample_logits=True, generator=generator,
    )
    s2_logits = model.decode_s2(context, sample_pre)
    ensure_before_deadline(deadline)
    sample_post = sample_from_logits(
        s2_logits[:, -1, :], temperature=temperature, top_k=top_k,
        top_p=top_p, sample_logits=True, generator=generator,
    )
    return sample_pre.squeeze(-1), sample_post.squeeze(-1)


def _append_autoregressive_tokens(
    generated_pre: torch.Tensor, generated_post: torch.Tensor,
    pre_buffer: torch.Tensor, post_buffer: torch.Tensor,
    sample_pre: torch.Tensor, sample_post: torch.Tensor,
    step: int, current_seq_len: int, max_context: int,
) -> None:
    generated_pre[:, step] = sample_pre
    generated_post[:, step] = sample_post
    if current_seq_len < max_context:
        pre_buffer[:, current_seq_len] = sample_pre
        post_buffer[:, current_seq_len] = sample_post
        return
    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
    pre_buffer[:, -1] = sample_pre
    post_buffer[:, -1] = sample_post


def _decode_autoregressive_samples(
    tokenizer: Any, x_token: Any, generated_pre: torch.Tensor,
    generated_post: torch.Tensor, total_seq_len: int, max_context: int,
    sample_count: int, deadline: float | None,
) -> np.ndarray:
    ensure_before_deadline(deadline)
    full_pre = torch.cat([x_token[0], generated_pre], dim=1)
    full_post = torch.cat([x_token[1], generated_post], dim=1)
    context_start = max(0, total_seq_len - max_context)
    input_tokens = [
        full_pre[:, context_start:total_seq_len].contiguous(),
        full_post[:, context_start:total_seq_len].contiguous(),
    ]
    decoded = tokenizer.decode(input_tokens, half=True)
    ensure_before_deadline(deadline)
    decoded = decoded.reshape(-1, sample_count, decoded.size(1), decoded.size(2))
    return decoded.cpu().numpy()


def auto_regressive_inference(
    tokenizer: Any, model: Any, x: torch.Tensor, x_stamp: torch.Tensor,
    y_stamp: torch.Tensor, max_context: int, pred_len: int, clip: float = 5,
    T: float = 1.0, top_k: int = 0, top_p: float = 0.99,
    sample_count: int = 5, verbose: bool = False,
    generator: torch.Generator | None = None, return_samples: bool = False,
    deadline: float | None = None,
) -> np.ndarray:
    with torch.no_grad():
        ensure_before_deadline(deadline)
        x = torch.clip(x, -clip, clip)
        state = _prepare_autoregressive_state(
            tokenizer, x, x_stamp, y_stamp, max_context, pred_len, sample_count,
        )
        (
            x_token, full_stamp, generated_pre, generated_post,
            pre_buffer, post_buffer, initial_seq_len, total_seq_len,
        ) = state
        ran = trange if verbose else range
        for i in ran(pred_len):
            ensure_before_deadline(deadline)
            current_seq_len = initial_seq_len + i
            input_tokens, current_stamp = _autoregressive_context(
                pre_buffer, post_buffer, full_stamp, current_seq_len, max_context,
            )
            sample_pre, sample_post = _sample_autoregressive_pair(
                model, input_tokens, current_stamp, T, top_k, top_p,
                generator, deadline,
            )
            _append_autoregressive_tokens(
                generated_pre, generated_post, pre_buffer, post_buffer,
                sample_pre, sample_post, i, current_seq_len, max_context,
            )
        samples = _decode_autoregressive_samples(
            tokenizer, x_token, generated_pre, generated_post, total_seq_len,
            max_context, sample_count, deadline,
        )
        if return_samples:
            return samples
        return np.mean(samples, axis=1)


def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    return time_df


class KronosPredictor:

    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):

        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose)
        preds = preds[:, -pred_len:, :]
        return preds

    def generate_paths(
        self, x: np.ndarray, x_stamp: np.ndarray, y_stamp: np.ndarray,
        pred_len: int, T: float, top_k: int, top_p: float,
        sample_count: int, sample_batch_size: int, seed: int,
        deadline: float | None = None,
    ) -> np.ndarray:
        """按小批次采样，避免一次生成全部路径造成显存峰值。"""
        batches = []
        for offset in range(0, sample_count, sample_batch_size):
            ensure_before_deadline(deadline)
            count = min(sample_batch_size, sample_count - offset)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed + offset)
            batch = self._generate_path_batch(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, count, generator, deadline)
            batches.append(batch)
        return np.concatenate(batches, axis=1)

    def _generate_path_batch(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, generator, deadline=None):
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.asarray(x_stamp, dtype=np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.asarray(y_stamp, dtype=np.float32)).to(self.device)
        samples = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count, False, generator, True, deadline)
        return samples[:, :, -pred_len:, :]

    def predict_paths(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series | pd.DatetimeIndex,
        y_timestamp: pd.Series | pd.DatetimeIndex,
        pred_len: int = 60,
        sample_count: int = 100,
        sample_batch_size: int = 20,
        seed: int | None = None,
        T: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 0,
        deadline: float | None = None,
    ) -> PredictionPaths:
        """生成真实采样路径，且不改变旧 ``predict`` 的均值路径行为。"""
        x, x_stamp, y_stamp, mean, std = self._prepare_prediction_input(df, x_timestamp, y_timestamp)
        self._validate_path_request(pred_len, sample_count, sample_batch_size, len(y_stamp[0]))
        sample_batch_size = min(sample_batch_size, sample_count)
        params_hash = sampling_params_hash(pred_len, sample_count, sample_batch_size, T, top_p, top_k)
        resolved_seed = self._resolve_seed(seed, x, y_stamp, params_hash)
        samples = self.generate_paths(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, sample_batch_size, resolved_seed, deadline)
        paths = samples.squeeze(0) * (std + 1e-5) + mean
        repaired, repaired_values = self._repair_paths(paths)
        timestamps = pd.DatetimeIndex(pd.to_datetime(y_timestamp))
        mean_df = pd.DataFrame(repaired.mean(axis=0), columns=self.price_cols + [self.vol_col, self.amt_vol], index=timestamps)
        flags = ("PATH_REPAIR_EXCESSIVE",) if repaired_values / max(1, repaired.size) > 0.01 else ()
        return PredictionPaths(repaired.astype(np.float32), mean_df, timestamps, sample_count, resolved_seed, self._object_id(self.model), self._object_id(self.tokenizer), params_hash, repaired_values, flags)

    def _prepare_prediction_input(self, df, x_timestamp, y_timestamp):
        if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in self.price_cols):
            raise ValueError("输入必须是包含 OHLC 列的 pandas DataFrame。")
        x_timestamp = pd.Series(x_timestamp) if isinstance(x_timestamp, pd.DatetimeIndex) else x_timestamp
        y_timestamp = pd.Series(y_timestamp) if isinstance(y_timestamp, pd.DatetimeIndex) else y_timestamp
        prepared = df.copy()
        if self.vol_col not in prepared:
            prepared[self.vol_col] = 0.0
        if self.amt_vol not in prepared:
            prepared[self.amt_vol] = prepared[self.vol_col] * prepared[self.price_cols].mean(axis=1)
        values = prepared[self.price_cols + [self.vol_col, self.amt_vol]].to_numpy(dtype=np.float32)
        if np.isnan(values).any():
            raise ValueError("输入数据包含 OHLCVA 空值。")
        mean, std = values.mean(axis=0), values.std(axis=0)
        normalized = np.clip((values - mean) / (std + 1e-5), -self.clip, self.clip)
        return normalized[None, :], calc_time_stamps(x_timestamp).to_numpy(dtype=np.float32)[None, :], calc_time_stamps(y_timestamp).to_numpy(dtype=np.float32)[None, :], mean, std

    @staticmethod
    def _validate_path_request(pred_len, sample_count, sample_batch_size, timestamp_count):
        if pred_len <= 0 or pred_len != timestamp_count:
            raise ValueError("pred_len 必须与预测时间戳数量一致且大于零。")
        if sample_count <= 0 or sample_batch_size <= 0:
            raise ValueError("采样数量和采样批大小必须大于零。")

    @staticmethod
    def _repair_paths(paths):
        repaired = np.array(paths, dtype=np.float64, copy=True)
        before = repaired.copy()
        repaired[:, :, 1] = np.maximum.reduce([repaired[:, :, 0], repaired[:, :, 1], repaired[:, :, 2], repaired[:, :, 3]])
        repaired[:, :, 2] = np.minimum.reduce([repaired[:, :, 0], repaired[:, :, 1], repaired[:, :, 2], repaired[:, :, 3]])
        repaired[:, :, 4:] = np.maximum(repaired[:, :, 4:], 0.0)
        return repaired, int(np.count_nonzero(repaired != before))

    @staticmethod
    def _object_id(obj):
        return str(getattr(obj, "name_or_path", type(obj).__name__))

    @staticmethod
    def _sampling_hash(T, top_p, top_k, sample_count, pred_len=60, sample_batch_size=20):
        return sampling_params_hash(pred_len, sample_count, sample_batch_size, T, top_p, top_k)

    def _resolve_seed(self, seed, x, y_stamp, params_hash):
        if seed is not None:
            return int(seed)
        payload = np.asarray(x).tobytes() + np.asarray(y_stamp).tobytes() + params_hash.encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0  # Fill missing volume with zeros
            df[self.amt_vol] = 0.0  # Fill missing amount with zeros
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df


    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """
        Perform parallel (batch) prediction on multiple time series. All series must have the same historical length and prediction length (pred_len).

        Args:
            df_list (List[pd.DataFrame]): List of input DataFrames, each containing price columns and optional volume/amount columns.
            x_timestamp_list (List[pd.DatetimeIndex or Series]): List of timestamps corresponding to historical data, length should match the number of rows in each DataFrame.
            y_timestamp_list (List[pd.DatetimeIndex or Series]): List of future prediction timestamps, length should equal pred_len.
            pred_len (int): Number of prediction steps.
            T (float): Sampling temperature.
            top_k (int): Top-k filtering threshold.
            top_p (float): Top-p (nucleus sampling) threshold.
            sample_count (int): Number of parallel samples per series, automatically averaged internally.
            verbose (bool): Whether to display autoregressive progress.

        Returns:
            List[pd.DataFrame]: List of prediction results in the same order as input, each DataFrame contains
                                `open, high, low, close, volume, amount` columns, indexed by corresponding `y_timestamp`.
        """
        # Basic validation
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        # Require all series to have consistent historical and prediction lengths for batch processing
        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)           # (B, seq_len, feat)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32) # (B, seq_len, time_feat)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32) # (B, pred_len, time_feat)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)
        # preds: (B, pred_len, feat)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs

