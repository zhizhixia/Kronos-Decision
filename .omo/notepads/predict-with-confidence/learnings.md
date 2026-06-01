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

# Learnings - Task 3: plot_distribution() + main() Pipeline

## Completed: 2026-06-02

### Key Implementation Details

1. **plot_distribution() Function** (lines 343-391)
   - Dual-panel matplotlib figure: upper (price + confidence band), lower (terminal price histogram)
   - Uses `gridspec_kw={"height_ratios": [3, 1]}` for panel sizing
   - Returns chart path for console output
   - Annotation box shows key stats: Current, Target, Return%, Up Prob, CI

2. **main() Pipeline** (lines 395-575)
   - Model load → Per-stock loop → Final summary table
   - Per-stock steps:
     1. Fetch data via `fetch_stock_data(code)`
     2. Extract x_raw (last LOOKBACK rows), x_ts, y_ts (future dates)
     3. **Normalization**: `x_norm = (x_raw - x_mean) / (x_std + 1e-5)` then `np.clip(x_norm, -CLIP, CLIP)`
     4. Compute timestamps via `calc_time_stamps(x_ts).values`
     5. Convert to torch tensors (float32), move to DEVICE
     6. Call `generate_distribution()` — returns (SAMPLE_COUNT, PRED_LEN, 6) normalized paths
     7. **Denormalization**: `paths_denorm = paths * (x_std + 1e-5) + x_mean`
     8. Call `analyze_distribution(paths_denorm, current_price, name, code)`
     9. Call `plot_distribution()` — saves chart
     10. Save `{code}_{name}_confidence.csv`
     11. Save `{code}_{name}_paths.csv` (50 columns: path_1..path_50)
     12. Print detailed stats block

3. **Critical Formula Match**
   - Normalization matches `KronosPredictor.predict` (L544-547):
     ```python
     x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
     x = (x - x_mean) / (x_std + 1e-5)
     x = np.clip(x, -self.clip, self.clip)
     ```
   - Denormalization applied per-path BEFORE `analyze_distribution`

4. **Output Files Per Stock** (3 files)
   - `{code}_{name}_confidence.csv`: columns = [date, pred_close_mean, pred_close_median, pred_close_std, pred_close_p10, pred_close_p90, up_probability]
   - `{code}_{name}_paths.csv`: 50 columns (path_1..path_50), indexed by future_dates
   - `{code}_{name}_confidence_chart.png`: dual-panel visualization

5. **DEVICE Auto-Detection**
   - Changed from hardcoded `"cuda:0"` to `"cuda:0" if torch.cuda.is_available() else "cpu"`
   - Falls back to CPU if CUDA not available

6. **calc_time_stamps Pattern**
   - Expects `pd.Series`, NOT `pd.DatetimeIndex`
   - Use `.reset_index(drop=True)` on timestamp Series
   - Returns 5-column DataFrame: (minute, hour, weekday, day, month)
   - Use `.values` to get numpy array for tensor conversion

### File Structure (Final)
- Lines 1-44: Imports + config
- Lines 47-88: `fetch_stock_data()`
- Lines 91-99: `generate_future_dates()`
- Lines 102-248: `generate_distribution()`
- Lines 251-340: `analyze_distribution()`
- Lines 343-391: `plot_distribution()` **NEW**
- Lines 395-575: `main()` **REPLACED**
- Total: 575 lines (from original 362 lines)

