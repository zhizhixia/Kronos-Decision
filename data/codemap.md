# data/

Kronos 增强版新增数据管道，为决策系统提供统一的 A 股数据接入层。

---

## 职责 (Responsibility)

提供可靠、可降级的 A 股历史行情数据与动态股票池，是决策系统（`decision/`）的基础数据依赖。

| 文件 | 核心职责 |
|------|----------|
| `fetcher.py` | 个股日 K 线获取：三级降级策略（akshare → baostock → 本地缓存），含重试机制与数据完整性校验 |
| `pool.py` | 沪深 300 动态成分股池：通过 akshare 获取最新成分列表，本地 CSV 缓存每周自动刷新 |

---

## 设计模式 (Design Patterns)

### fetcher.py

| 模式 | 体现位置 | 说明 |
|------|----------|------|
| **Chain of Responsibility（职责链）** | `fetch_daily()` 第 58–85 行 | 三级降级链：缓存（有效）→ akshare（主）→ baostock（备）→ 缓存（过期）。每级失败后静默 fallthrough 到下一级 |
| **Template Method（模板方法）** | `_fetch_with_retry()` 第 89–101 行 | 统一的带指数退避重试骨架，`_fetch_akshare` / `_fetch_baostock` 作为策略参数注入 |
| **Strategy（策略）** | `_fetch_akshare` vs `_fetch_baostock` | 两个数据源实现不同 API 但输出统一格式（`_normalize_dataframe` 归一化），可互换 |
| **Cache-Aside（旁路缓存）** | `_load_cache` / `_save_cache` / `_is_cache_fresh` | 读取前查缓存（有效直接返回），写入后更新缓存；带 TTL + 收盘时效性双重过期判断 |
| **Retry with Backoff** | `_fetch_with_retry()` | 3 次重试，退避间隔 [2s, 4s, 8s] 由 `config.data.retry_backoff_seconds` 控制 |
| **Adapter（适配器）** | `_normalize_dataframe()` 第 141–168 行 | 统一 akshare（中文列名 "日期"）和 baostock（英文列名 "date"）的 DataFrame 格式为统一内部规约 |

### pool.py

| 模式 | 体现位置 | 说明 |
|------|----------|------|
| **Single-Function Module（函数式模块）** | 全部函数为模块级（无类） | 无状态设计，`get_hs300_pool()` 作为唯一公共入口，内部缓存透明 |
| **Cache-Aside（旁路缓存）** | `_load_cache` / `_save_cache` / `_is_fresh` | 7 天 TTL 的文件缓存；`refresh_pool()` 提供强制刷新接口 |
| **Failover（故障降级）** | `get_hs300_pool()` 第 38–40 行 | akshare 获取失败时静默返回过期缓存，仅首次全链路失败才抛异常 |

---

## 数据流 (Data Flow)

### fetcher.py — `fetch_daily(stock_code)`

```
调用者（如 decision/engine.py）
  │
  ├─ [1] _load_cache ──────────→ 缓存命中且 TTL 内 & 今日未收盘 ──→ 直接返回 DataFrame
  │
  └─ [2] _fetch_with_retry(_fetch_akshare)
  │       │
  │       ├─ akshare.stock_zh_a_hist() ──→ DataFrame
  │       ├─ _normalize_dataframe ──────→ 统一列名、日期索引、数值类型
  │       ├─ _validate ─────────────────→ 列完整性、NaN 比例、高低价合理性、最少行数(50)
  │       ├─ _save_cache ───────────────→ data/cache/{code}.csv
  │       └─ return DataFrame
  │
  └─ [3] _fetch_with_retry(_fetch_baostock)     ← akshare 失败后降级
  │       │
  │       ├─ bs.login() + query_history_k_data_plus()
  │       ├─ (同上 _normalize → _validate → _save_cache)
  │       └─ return DataFrame
  │
  └─ [4] _load_cache (过期缓存)  ← baostock 也失败后最终降级
          │
          ├─ warnings.warn("数据源不可用，使用缓存数据")
          └─ return DataFrame（可能不是最新）
               │
               └─ 全部失败且无缓存 → raise DataSourceError
```

**缓存过期判断逻辑**（`_is_cache_fresh`）：
1. 文件存在且 mtime < `ttl_hours`（默认 24h）
2. 今日 16:00 之后 → 所有当日前的缓存立即过期（保证收盘后获取最新数据）

### pool.py — `get_hs300_pool()`

```
调用者（如 decision/engine.py, webui）
  │
  ├─ _load_cache ─────────→ pool_hs300.csv 存在且 7 天内 → 返回 {code: name}
  │
  └─ _fetch_from_akshare ─→ ak.index_stock_cons("000300")
  │       │                  → 提取 "品种代码"/"品种名称" 列
  │       ├─ _save_cache ──→ 写入 pool_hs300.csv
  │       └─ return dict
  │
  └─ akshare 失败且有过期缓存 → 返回过期缓存（降级）
      否则 → raise DataSourceError

强制刷新（webui 设置页调用）：
  refresh_pool() → _fetch_from_akshare → _save_cache → return dict
```

---

## 集成点 (Integration Points)

### 对外依赖

| 外部模块 | 用途 | 文件 |
|----------|------|------|
| `decision.config.get_config()` | 读取 `DataConfig`：`cache_dir`、`cache_ttl_hours`、`retry_max`、`retry_backoff_seconds` | fetcher.py + pool.py |
| `decision.errors.DataSourceError` | 数据源不可用时抛出的自定义异常 | fetcher.py + pool.py |
| `decision.errors.DataValidationError` | 数据校验失败时抛出的自定义异常 | fetcher.py |
| akshare (第三方) | 主数据源：个股日 K + 指数成分股 | fetcher.`_fetch_akshare` + pool.`_fetch_from_akshare` |
| baostock (第三方) | 备用数据源：个股日 K | fetcher.`_fetch_baostock` |
| pandas | 数据容器与 CSV 持久化 | 两文件均使用 |

### 对内接口

| 函数/方法 | 签名 | 消费者 | 说明 |
|-----------|------|--------|------|
| `DataFetcher.fetch_daily(code)` | `(str) → pd.DataFrame` | `decision/engine.py`、`examples/` | 获取个股 OHLCV 数据 |
| `DataFetcher.__init__()` | `() → None` | 调用者实例化 | 从配置读取缓存路径与重试参数 |
| `get_hs300_pool()` | `() → dict[str, str]` | `decision/engine.py`、`webui/` | 获取沪深 300 成分股字典 |
| `refresh_pool()` | `() → dict[str, str]` | `webui/`（设置页面） | 强制刷新股票池 |

### 文件接口

| 文件路径 | 读写方 | 格式 |
|----------|--------|------|
| `data/cache/{stock_code}.csv` | `DataFetcher._save_cache` / `_load_cache` | CSV，含 `date,open,high,low,close,volume,amount` |
| `data/cache/pool_hs300.csv` | pool.`_save_cache` / `_load_cache` | CSV，含 `code,name` 两列 |

### 关键假设与约束

1. **股票代码格式**：6 位纯数字字符串（如 `"600519"`），不含交易所前缀
2. **baostock 代码转换**：`fetcher._fetch_baostock` 根据首位数字推断交易所：`5`/`6`/`9` 开头 → `sh.`，其余 → `sz.`
3. **收盘时效**：每日 16:00 后标记缓存过期，确保收盘后首次调用触发网络请求获取最新数据
4. **最小数据量**：`_validate` 要求至少 50 行有效数据（约 2.5 个月交易日），低于此阈值视为无效
5. **价格合理性**：`high < low` 的行数超过 5% 时整批数据视为无效并抛 `DataValidationError`
6. **决定系统`config.yaml`的`DataConfig`段**可控制所有数据获取参数，实现零代码配置变更
