# Learnings - Task 1: generate_distribution()

## Completed: 2026-06-02

### Key Implementation Details

1. **Chunking Strategy**
   - Process paths_per_batch=10 samples at a time
   - Loop until sample_count=50 total samples collected
   - Concatenate chunks along axis 0 (sample dimension)

2. **Critical Code Changes from Original uto_regressive_inference**
   - NO 
p.mean(preds, axis=1) - preserves all sample paths
   - Added outer chunking loop with emaining counter
   - Return shape: (sample_count, pred_len, 6) via slicing ll_paths[:, -pred_len:, :]

3. **Model State Management**
   - 	okenizer.eval() and model.eval() called INSIDE generate_distribution()
   - 	orch.manual_seed(42) for reproducibility
   - with torch.no_grad(): context for inference

4. **Import Pattern**
   - rom model.kronos import calc_time_stamps, sample_from_logits
   - Standard sys.path.insert(0, ...) pattern for project root

### File Structure
- Lines 1-27: Docstring + imports
- Lines 29-30: Chinese font config
- Lines 32-37: Model imports
- Lines 39-49: Config constants
- Lines 52-82: etch_stock_data() (copied from predict_my_stocks.py)
- Lines 85-93: generate_future_dates() (copied from predict_my_stocks.py)
- Lines 97-243: generate_distribution() (core function)
- Lines 247-265: main() placeholder

### Bug Fix: Dimension Mismatch (2026-06-02)
- **Issue**: Input `x`, `x_stamp`, `y_stamp` are 2D `(seq_len, feat)` but code expected 3D `(1, seq_len, feat)`
- **Error**: `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)`
- **Fix**: Added `x.unsqueeze(0)`, `x_stamp.unsqueeze(0)`, `y_stamp.unsqueeze(0)` at start of `with torch.no_grad():` block
- **Why**: Original `auto_regressive_inference` receives 3D input from `KronosPredictor.generate()` via `x[np.newaxis, :]`

### Dependencies for Tasks 2-3
- Task 2 will add analyze_distribution() for confidence intervals
- Task 3 will add plot_distribution() and per-stock loop

