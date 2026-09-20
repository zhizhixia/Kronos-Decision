/**
 * Kronos 决策报告 v2 页面脚本。
 * 约束：不通过 innerHTML 渲染 API 数据；所有动态内容使用 textContent/createElement。
 */
(function (global) {
  "use strict";

  const WB = global.KronosWorkbench;
  const RECENT_KEY = "kronos_report_recent";
  const MAX_RECENT = 8;

  const ACTION_LABELS = {
    ADD: "增持",
    HOLD: "持有",
    REDUCE: "减持",
    AVOID: "回避",
    INSUFFICIENT_EVIDENCE: "证据不足",
  };

  const ACTION_SUB = {
    ADD: "ADD · 20日主期限正式证据支持增持",
    HOLD: "HOLD · 证据通过但未达调仓阈值",
    REDUCE: "REDUCE · 证据支持降低现有持仓",
    AVOID: "AVOID · 证据提示回避该股票",
    INSUFFICIENT_EVIDENCE: "INSUFFICIENT_EVIDENCE · 门禁未通过",
  };

  const RUN_STATUSES = new Set(["SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT"]);
  const EVIDENCE_STATUSES = new Set(["RESEARCH_ONLY", "QUALIFIED", "INSUFFICIENT", "STALE", "INVALID"]);
  const ACTION_PERMISSIONS = new Set(["NONE", "CONDITIONAL_REFERENCE", "INSUFFICIENT_EVIDENCE"]);
  const ACTIONS_WITH_REFERENCE = new Set(["ADD", "HOLD", "REDUCE", "AVOID"]);
  const LEGACY_SIGNAL_BY_ACTION = { ADD: "BUY", HOLD: "HOLD", REDUCE: "SELL", AVOID: "SELL" };

  const GATE_LABELS = {
    DATA_COMPLETE: "数据截止完整交易日",
    NO_QUALITY_ERROR: "无未解决质量错误",
    INTERVAL_COVERAGE: "80%区间实际覆盖率75%-85%",
    PROBABILITY_ECE: "上涨概率ECE≤0.08",
    RANK_IC: "样本外RankIC均值>0",
    BOOTSTRAP_RANK_IC: "Bootstrap RankIC为正概率≥80%",
    NET_EXCESS_RETURN: "扣费年化超额收益>0",
    INFORMATION_RATIO: "信息比率≥0.5",
    WINDOW_STABILITY: "60%以上12个月窗口正超额",
    DRAWDOWN: "回撤恶化不超过5个百分点",
    VERSION_MATCH: "评估/模型/配置/规则/数据版本匹配",
    FORMAL_PROTOCOL: "完整正式研究协议（含沪深300全收益基准）",
    ONLINE_AS_OF_MATCH: "在线报告与评估使用同一完整交易日",
    FRESH_ARTIFACT: "评估产物30天内",
  };

  const GATE_RECOVERY = {
    DATA_COMPLETE: "等待完整收盘日线并修复数据截止日质量错误",
    NO_QUALITY_ERROR: "修复数据质量标记后重新生成报告",
    INTERVAL_COVERAGE: "扩充滚动样本外校准并重新评估区间覆盖率",
    PROBABILITY_ECE: "扩充滚动样本外校准并重新评估上涨概率校准误差",
    RANK_IC: "完成足够长的样本外评估，确认主期限 RankIC 为正",
    BOOTSTRAP_RANK_IC: "增加独立锚点后重新进行区块 Bootstrap 检验",
    NET_EXCESS_RETURN: "在扣费回测后取得正年化超额收益",
    INFORMATION_RATIO: "在扣费回测后达到信息比率门槛",
    WINDOW_STABILITY: "增加滚动12个月观察窗口并达到稳定性门槛",
    DRAWDOWN: "降低相对基准的回撤恶化后重新评估",
    VERSION_MATCH: "使用与评估工件完全一致的模型、配置、规则、采样参数和数据版本",
    FORMAL_PROTOCOL: "导入覆盖正式区间的 H00300 全收益基准，并完成2017年起、100路径、完整股票池的周度评估",
    ONLINE_AS_OF_MATCH: "在评估截止日重新生成在线报告，或以最新完整交易日重跑正式评估",
    FRESH_ARTIFACT: "重新完成30天内的正式评估",
  };

  let stockList = null;
  let stockListPromise = null;
  let isRequesting = false;
  let elapsedTimer = null;
  let lastRequest = null;
  let suggestionIndex = -1;

  function getElement(id) {
    return document.getElementById(id);
  }

  function show(element) {
    if (element) {
      element.hidden = false;
    }
  }

  function hide(element) {
    if (element) {
      element.hidden = true;
    }
  }

  function setClass(element, baseClass, actionClass) {
    if (!element) {
      return;
    }
    element.className = baseClass + " " + actionClass;
  }

  function validateCode(code) {
    return /^\d{6}$/.test(code);
  }

  function readRecent() {
    try {
      const raw = global.localStorage.getItem(RECENT_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.slice(0, MAX_RECENT) : [];
    } catch (err) {
      return [];
    }
  }

  function writeRecent(items) {
    try {
      global.localStorage.setItem(RECENT_KEY, JSON.stringify(items.slice(0, MAX_RECENT)));
    } catch (err) {
      // ignore storage errors
    }
  }

  function pushRecent(code, name) {
    const items = readRecent().filter(function (item) {
      return item.code !== code;
    });
    items.unshift({ code: code, name: name || "" });
    writeRecent(items);
  }

  function loadStockList() {
    if (stockListPromise) {
      return stockListPromise;
    }
    stockListPromise = WB.getJson("/api/stock-list", 15000).then(function (result) {
      if (result.ok && result.payload.stocks) {
        stockList = result.payload.stocks;
        return stockList;
      }
      stockList = [];
      return stockList;
    });
    return stockListPromise;
  }

  function findStock(code) {
    if (!stockList) {
      return null;
    }
    return stockList.find(function (s) {
      return s.code === code;
    }) || null;
  }

  function setInputExpanded(expanded) {
    const input = getElement("stock-code");
    if (input) {
      input.setAttribute("aria-expanded", String(expanded));
    }
  }

  function setActiveDescendant(id) {
    const input = getElement("stock-code");
    if (input) {
      input.setAttribute("aria-activedescendant", id || "");
    }
  }

  function updateSuggestions(query) {
    const list = getElement("stock-suggestions");
    const input = getElement("stock-code");
    if (!list) {
      return;
    }
    WB.clearChildren(list);
    suggestionIndex = -1;
    setActiveDescendant("");
    if (!stockList || query.length < 1) {
      hide(list);
      setInputExpanded(false);
      return;
    }
    const q = query.trim();
    const matches = stockList
      .filter(function (s) {
        return s.code.indexOf(q) === 0 || s.name.indexOf(q) !== -1;
      })
      .slice(0, 8);
    if (matches.length === 0) {
      hide(list);
      setInputExpanded(false);
      return;
    }
    matches.forEach(function (s, index) {
      const li = WB.appendChild(list, "li", {
        role: "option",
        id: "stock-suggestion-" + index,
        className: "suggestion-item",
        ariaSelected: "false",
      });
      WB.appendChild(li, "span", { className: "suggestion-code" }, [s.code]);
      WB.appendChild(li, "span", { className: "suggestion-name" }, [s.name]);
      li.addEventListener("click", function () {
        selectSuggestion(s.code, s.name);
      });
    });
    setInputExpanded(true);
    show(list);
  }

  function selectSuggestion(code, name) {
    const input = getElement("stock-code");
    if (input) {
      input.value = code;
    }
    WB.setText(getElement("stock-name-display"), name || (findStock(code) && findStock(code).name) || "");
    hide(getElement("stock-suggestions"));
    setInputExpanded(false);
    setActiveDescendant("");
    suggestionIndex = -1;
  }

  function moveSuggestion(direction) {
    const list = getElement("stock-suggestions");
    if (!list || list.hidden) {
      return;
    }
    const items = list.querySelectorAll("li[role='option']");
    if (items.length === 0) {
      return;
    }
    if (suggestionIndex >= 0 && suggestionIndex < items.length) {
      items[suggestionIndex].setAttribute("aria-selected", "false");
      items[suggestionIndex].classList.remove("suggestion-active");
    }
    suggestionIndex += direction;
    if (suggestionIndex < 0) {
      suggestionIndex = items.length - 1;
    } else if (suggestionIndex >= items.length) {
      suggestionIndex = 0;
    }
    const active = items[suggestionIndex];
    active.setAttribute("aria-selected", "true");
    active.classList.add("suggestion-active");
    active.scrollIntoView({ block: "nearest" });
    setActiveDescendant(active.id);
  }

  function applyActiveSuggestion() {
    const list = getElement("stock-suggestions");
    if (!list || list.hidden) {
      return false;
    }
    const items = list.querySelectorAll("li[role='option']");
    if (suggestionIndex >= 0 && suggestionIndex < items.length) {
      items[suggestionIndex].click();
      return true;
    }
    return false;
  }

  function renderRecent() {
    const bar = getElement("recent-bar");
    const list = getElement("recent-list");
    if (!bar || !list) {
      return;
    }
    WB.clearChildren(list);
    const items = readRecent();
    if (items.length === 0) {
      hide(bar);
      return;
    }
    items.forEach(function (item) {
      const btn = WB.appendChild(list, "button", {
        type: "button",
        className: "recent-chip",
        ariaLabel: "加载 " + item.code,
      }, [item.code]);
      btn.addEventListener("click", function () {
        selectSuggestion(item.code, item.name);
        generateReport();
      });
    });
    show(bar);
  }

  function updateStatus(message, level) {
    const card = getElement("status-card");
    const msg = getElement("status-message");
    if (!card || !msg) {
      return;
    }
    WB.setText(msg, message);
    card.className = "report-status wb-card status-level-" + level;
    show(card);
  }

  function startElapsedTimer(startedAt) {
    const badge = getElement("elapsed-badge");
    const text = getElement("elapsed-text");
    if (badge) {
      show(badge);
    }
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
    }
    function tick() {
      const seconds = (Date.now() - startedAt) / 1000;
      WB.setText(text, seconds.toFixed(1) + "s");
    }
    tick();
    elapsedTimer = setInterval(tick, 100);
  }

  function stopElapsedTimer() {
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }

  async function generateReport(mode) {
    mode = mode || "model";
    const codeInput = getElement("stock-code");
    const btn = getElement("generate-btn");
    const baselineBtn = getElement("baseline-btn");
    const code = codeInput ? codeInput.value.trim() : "";

    if (!validateCode(code)) {
      updateStatus("股票代码必须是六位数字", "error");
      return;
    }
    if (isRequesting) {
      return;
    }

    isRequesting = true;
    if (btn) {
      btn.disabled = true;
    }
    if (baselineBtn) {
      baselineBtn.disabled = true;
    }
    hide(getElement("stock-suggestions"));
    setInputExpanded(false);
    updateStatus(mode === "baseline" ? "正在生成无模型基线..." : "正在生成报告...", "warn");
    const startedAt = Date.now();
    startElapsedTimer(startedAt);

    const includePaths = getElement("show-paths")?.checked || false;
    const stock = findStock(code);
    const name = stock ? stock.name : "";

    lastRequest = {
      stock_code: code,
      portfolio_id: "default",
      include_display_paths: includePaths,
      mode: mode,
    };

    const result = await WB.postJson("/api/v2/decision-report", lastRequest, 120000);

    stopElapsedTimer();
    isRequesting = false;
    if (btn) {
      btn.disabled = false;
    }
    if (baselineBtn) {
      baselineBtn.disabled = false;
    }

    const elapsedSeconds = (Date.now() - startedAt) / 1000;
    renderContract(result.payload);

    if (!result.ok) {
      const error = result.error || {};
      updateStatus(error.message || "生成报告失败", "error");
      renderError(error, elapsedSeconds);
      return;
    }

    pushRecent(code, name || result.payload.stock?.name || "");
    renderRecent();
    renderReport(result.payload, elapsedSeconds);
  }

  function retryLastRequest() {
    if (!lastRequest) {
      return;
    }
    const input = getElement("stock-code");
    if (input) {
      input.value = lastRequest.stock_code;
    }
    generateReport(lastRequest.mode || "model");
  }

  function renderError(error, elapsedSeconds) {
    const status = getElement("status-message");
    if (status) {
      WB.setText(status, (error.message || "生成报告失败") + " (" + elapsedSeconds.toFixed(1) + "s)");
    }

    const errorCard = getElement("error-card");
    const errorTitle = getElement("error-title-text");
    const errorCode = getElement("error-code");
    const errorMessage = getElement("error-message");
    const errorDetails = getElement("error-details");
    const retryBtn = getElement("error-retry-btn");

    if (errorCard) {
      show(errorCard);
    }
    WB.setText(errorTitle, error.retryable ? "请求失败，可重试" : "请求失败");
    WB.setText(errorCode, error.code ? "错误码：" + error.code : "");
    WB.setText(errorMessage, error.message || "生成报告失败");

    if (errorDetails) {
      WB.clearChildren(errorDetails);
      if (error.details && typeof error.details === "object" && Object.keys(error.details).length > 0) {
        Object.entries(error.details).forEach(function (_ref) {
          const key = _ref[0];
          const value = _ref[1];
          WB.appendChild(errorDetails, "div", { className: "error-detail-row" }, [
            key + "：" + (value == null ? "" : String(value)),
          ]);
        });
      }
    }

    if (retryBtn) {
      retryBtn.hidden = !error.retryable;
      retryBtn.disabled = false;
    }

    hide(getElement("action-card"));
    hide(getElement("horizon-card"));
    hide(getElement("chart-card"));
    hide(getElement("provenance-card"));
    hide(getElement("gate-card"));
    hide(getElement("impact-card"));
    hide(getElement("market-card"));
    hide(getElement("fundamental-card"));
  }

  function hideErrorCard() {
    hide(getElement("error-card"));
    const retryBtn = getElement("error-retry-btn");
    if (retryBtn) {
      retryBtn.disabled = false;
    }
  }

  function renderContract(payload) {
    const source = payload && typeof payload === "object" ? payload : {};
    WB.setText(getElement("run-status"), source.run_status || "未知");
    WB.setText(getElement("evidence-status"), source.evidence_status || "未知");
    WB.setText(getElement("action-permission"), source.action_permission || "NONE");
  }

  function hasValidReferenceContract(payload) {
    return Boolean(
      payload &&
      RUN_STATUSES.has(payload.run_status) &&
      payload.run_status === "SUCCEEDED" &&
      EVIDENCE_STATUSES.has(payload.evidence_status) &&
      payload.evidence_status === "QUALIFIED" &&
      ACTION_PERMISSIONS.has(payload.action_permission) &&
      payload.action_permission === "CONDITIONAL_REFERENCE"
    );
  }

  function safeRecommendation(payload) {
    const recommendation = payload && typeof payload.recommendation === "object"
      ? payload.recommendation
      : {};
    const action = typeof recommendation.action === "string"
      ? recommendation.action.toUpperCase()
      : "";
    if (!hasValidReferenceContract(payload) || !ACTIONS_WITH_REFERENCE.has(action)) {
      return {
        action: "INSUFFICIENT_EVIDENCE",
        legacy_signal: null,
        reason_codes: ["INSUFFICIENT_EVIDENCE"],
      };
    }
    return Object.assign({}, recommendation, {
      action: action,
      legacy_signal: LEGACY_SIGNAL_BY_ACTION[action],
    });
  }

  function renderReport(payload, elapsedSeconds) {
    payload = payload && typeof payload === "object" ? payload : {};
    renderContract(payload);
    const recommendation = safeRecommendation(payload);
    hideErrorCard();
    updateStatus("完成，耗时 " + elapsedSeconds.toFixed(1) + "s", "ok");
    WB.setText(getElement("stock-name-display"), payload.stock?.name || "");

    renderAction(recommendation, payload.portfolio_impact);
    renderHorizons(payload.horizons);
    renderChart(payload.chart_data, payload.display_paths);
    renderProvenance(payload.data_provenance, payload.model_provenance);
    renderGate(payload.evidence_gate);
    renderImpact(payload.portfolio_impact, recommendation.action);
    renderMarket(payload.market_context);
    renderFundamentals(payload.fundamentals, payload.events);
  }

  function renderAction(recommendation, portfolioImpact) {
    const action = recommendation?.action || "INSUFFICIENT_EVIDENCE";
    const card = getElement("action-card");
    const root = getElement("action-root");
    show(card);
    setClass(root, "wb-action", "wb-action-" + action);

    WB.setText(getElement("action-value"), ACTION_LABELS[action] || action);
    WB.setText(getElement("action-sub"), ACTION_SUB[action] || action);
    WB.setText(getElement("legacy-signal"), recommendation?.legacy_signal || "--");
    WB.setText(getElement("hold-status"), portfolioImpact?.is_held ? "已持有" : "未持有");

    const reasonList = getElement("reason-list");
    WB.clearChildren(reasonList);
    const codes = recommendation?.reason_codes || [];
    if (codes.length > 0) {
      codes.forEach(function (code) {
        WB.appendChild(reasonList, "span", { className: "reason-chip" }, [code]);
      });
    }

    const disclaimer = getElement("action-disclaimer");
    if (action === "INSUFFICIENT_EVIDENCE") {
      show(disclaimer);
    } else {
      hide(disclaimer);
    }
  }

  function renderHorizons(horizons) {
    const card = getElement("horizon-card");
    const grid = getElement("horizon-grid");
    show(card);
    WB.clearChildren(grid);

    ["5", "20", "60"].forEach(function (horizon) {
      const stats = horizons && horizons[horizon] ? horizons[horizon] : {};
      const item = WB.appendChild(grid, "div", { className: "horizon-card" });
      WB.appendChild(item, "div", { className: "horizon-title" }, [horizon + " 日"]);
      renderKv(item, "q05", WB.formatPercent(stats.q05));
      renderKv(item, "q50（中位）", WB.formatPercent(stats.q50));
      renderKv(item, "q95", WB.formatPercent(stats.q95));
      renderKv(item, "上涨概率", WB.formatPercent(stats.up_probability));
      renderKv(item, "预测回撤", WB.formatPercent(stats.predicted_drawdown));
      renderKv(item, "校准状态", stats.calibration_status || "--");
    });
  }

  function renderKv(parent, key, value) {
    const row = WB.appendChild(parent, "div", { className: "wb-kv" });
    WB.appendChild(row, "span", { className: "wb-kv-key" }, [key]);
    WB.appendChild(row, "span", { className: "wb-kv-value" }, [value]);
  }

  function createIcon(useHref, className) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", className);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", useHref);
    svg.appendChild(use);
    return svg;
  }

  function renderGate(gate) {
    const card = getElement("gate-card");
    const summary = getElement("gate-summary");
    const list = getElement("gate-list");
    const recovery = getElement("gate-recovery");
    const recoveryText = getElement("gate-recovery-text");
    show(card);
    WB.clearChildren(list);

    if (!gate || !gate.checks) {
      WB.clearChildren(summary);
      WB.appendChild(summary, "div", { className: "gate-passed" }, ["无正式评估产物，门禁全部未通过"]);
      WB.appendChild(list, "div", { className: "gate-empty" }, ["请先完成正式评估并生成评估产物。"]);
      show(recovery);
      WB.setText(recoveryText, "恢复条件：完成 Qlib 样本外评估并生成 30 天内的评估产物。");
      return;
    }

    const passedCount = Object.values(gate.checks).filter(Boolean).length;
    const totalCount = Object.keys(gate.checks).length;
    WB.clearChildren(summary);
    WB.appendChild(summary, "div", { className: "gate-passed" }, ["通过 " + passedCount + " / " + totalCount + " 项"]);

    const entries = Object.entries(gate.checks).sort(function (a, b) {
      return Number(a[1]) - Number(b[1]);
    });

    entries.forEach(function (_ref) {
      const code = _ref[0];
      const passed = _ref[1];
      const row = WB.appendChild(list, "div", { className: "gate-row" });
      WB.appendChild(row, "span", { className: "gate-label" }, [GATE_LABELS[code] || code]);
      const status = WB.appendChild(row, "span", {
        className: passed ? "gate-status gate-pass" : "gate-status gate-fail",
      });
      const iconHref = passed ? "#icon-check" : "#icon-x";
      status.appendChild(createIcon(iconHref, "icon gate-icon"));
      status.appendChild(WB.textNode(passed ? "通过" : "未通过"));
    });

    if (!gate.passed) {
      show(recovery);
      const failed = gate.failed_codes || [];
      const parts = failed.map(function (code) {
        return (GATE_LABELS[code] || code) + "：" + (GATE_RECOVERY[code] || "修复该项后重新评估");
      });
      WB.setText(recoveryText, "未通过门禁：" + parts.join("；") + "。恢复后才会输出增持/减持/回避。");
    } else {
      hide(recovery);
    }
  }

  function renderImpact(portfolioImpact, action) {
    const card = getElement("impact-card");
    const body = getElement("impact-body");
    show(card);
    WB.clearChildren(body);
    renderKv(body, "组合 ID", portfolioImpact?.portfolio_id || "--");
    renderKv(body, "是否持仓", portfolioImpact?.is_held ? "是" : "否");
    renderKv(body, "可参与调仓", portfolioImpact?.eligible_for_rebalance ? "是" : "否");
    const hint = action === "INSUFFICIENT_EVIDENCE"
      ? "证据不足，当前不生成调仓建议。"
      : "请结合风控与组合目标独立判断是否执行。";
    WB.appendChild(body, "p", { className: "wb-note" }, [hint]);
  }

  function renderMarket(context) {
    const card = getElement("market-card");
    const body = getElement("market-body");
    show(card);
    WB.clearChildren(body);

    if (!context || !context.available) {
      WB.appendChild(body, "p", { className: "wb-note" }, [
        "市场背景数据不可用（" + (context?.reason || "unknown") + "）。此信息不改变量化动作。",
      ]);
      return;
    }

    const state = context.market_state || {};
    renderKv(body, "指数趋势", state.trend === "above_ma" ? "均线之上" : "均线之下");
    renderKv(body, "年化波动率", WB.formatPercent(state.annualized_volatility));
    renderKv(body, "波动状态", state.volatility_state || "--");

    const sectors = context.sector_strength?.sectors || {};
    Object.entries(sectors).slice(0, 5).forEach(function (_ref) {
      const name = _ref[0];
      const value = _ref[1];
      renderKv(body, name, WB.formatPercent(value));
    });
    WB.appendChild(body, "p", { className: "wb-note" }, [context.disclaimer || ""]);
  }

  function renderFundamentals(fundamentals, events) {
    const card = getElement("fundamental-card");
    const body = getElement("fundamental-body");
    show(card);
    WB.clearChildren(body);

    if (!fundamentals || !fundamentals.available) {
      WB.appendChild(body, "p", { className: "wb-note" }, [
        "基本面数据不可用（" + (fundamentals?.reason || "unknown") + "）。此信息不改变量化动作。",
      ]);
    } else {
      const items = fundamentals.items || {};
      if (Object.keys(items).length === 0) {
        WB.appendChild(body, "p", { className: "wb-note" }, ["暂无基本面详情。"]);
      } else {
        Object.entries(items).forEach(function (_ref) {
          const key = _ref[0];
          const value = _ref[1];
          renderKv(body, key, value == null ? "--" : String(value));
        });
      }
      WB.appendChild(body, "p", { className: "wb-note" }, [fundamentals.disclaimer || ""]);
    }

    const eventList = events?.events || [];
    if (eventList.length > 0) {
      WB.appendChild(body, "h4", { className: "fundamental-subtitle" }, ["近期事件"]);
      eventList.slice(0, 5).forEach(function (event) {
        renderKv(body, event.date || "--", event.title || event.type || "--");
      });
    }
    WB.appendChild(body, "p", { className: "wb-note" }, [events?.disclaimer || ""]);
  }

  function renderProvenance(dataProvenance, modelProvenance) {
    const card = getElement("provenance-card");
    const grid = getElement("provenance-grid");
    show(card);
    WB.clearChildren(grid);

    const provenance = dataProvenance || {};
    const sampling = modelProvenance || {};
    const pairs = [
      ["数据截止日", WB.formatDate(provenance.as_of)],
      ["数据源", provenance.source || "--"],
      ["新鲜度", provenance.is_stale ? "过期" : "正常"],
      ["模型", sampling.model_id || "--"],
      ["采样种子", sampling.seed != null ? String(sampling.seed) : "--"],
      ["采样数量", sampling.sample_count != null ? String(sampling.sample_count) : "--"],
      ["采样参数哈希", sampling.sampling_params_hash || "--"],
      ["路径修复比例", sampling.repair_ratio != null ? (sampling.repair_ratio * 100).toFixed(2) + "%" : "--"],
    ];
    pairs.forEach(function (pair) {
      const item = WB.appendChild(grid, "div", { className: "provenance-item" });
      WB.appendChild(item, "div", { className: "provenance-label" }, [pair[0]]);
      WB.appendChild(item, "div", { className: "provenance-value" }, [pair[1]]);
    });
  }

  function extractHistorySeries(history) {
    return {
      dates: history.map(function (bar) { return bar.timestamp; }),
      open: history.map(function (bar) { return bar.open; }),
      high: history.map(function (bar) { return bar.high; }),
      low: history.map(function (bar) { return bar.low; }),
      close: history.map(function (bar) { return bar.close; }),
      volume: history.map(function (bar) { return bar.volume; }),
    };
  }

  function buildCandlestickTrace(series) {
    return {
      x: series.dates,
      open: series.open,
      high: series.high,
      low: series.low,
      close: series.close,
      type: "candlestick",
      name: "历史 K 线",
      xaxis: "x",
      yaxis: "y",
      increasing: { line: { color: "#d93026" }, fillcolor: "#d93026" },
      decreasing: { line: { color: "#238636" }, fillcolor: "#238636" },
    };
  }

  function buildPathTraces(forecastDates, displayPaths) {
    const traces = [];
    if (!Array.isArray(displayPaths) || displayPaths.length === 0) {
      return traces;
    }
    displayPaths.slice(0, 20).forEach(function (path) {
      const closeSeries = path.map(function (point) {
        return point[3];
      });
      traces.push({
        x: forecastDates,
        y: closeSeries,
        type: "scatter",
        mode: "lines",
        line: { color: "rgba(37, 99, 235, 0.15)", width: 1 },
        name: "真实路径",
        showlegend: false,
        xaxis: "x",
        yaxis: "y",
        hoverinfo: "skip",
      });
    });
    return traces;
  }

  function buildQuantileTraces(forecastDates, q05, q50, q95) {
    return [
      {
        x: forecastDates.concat(forecastDates.slice().reverse()),
        y: q95.concat(q05.slice().reverse()),
        fill: "toself",
        fillcolor: "rgba(37, 99, 235, 0.12)",
        line: { color: "transparent" },
        name: "q05-q95 区间",
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y",
        hoverinfo: "skip",
      },
      {
        x: forecastDates,
        y: q50,
        type: "scatter",
        mode: "lines",
        line: { color: "#2563eb", width: 2, dash: "dash" },
        name: "q50 中位",
        xaxis: "x",
        yaxis: "y",
      },
    ];
  }

  function buildVolumeTrace(dates, volumes) {
    return {
      x: dates,
      y: volumes,
      type: "bar",
      name: "成交量",
      marker: { color: "rgba(143, 170, 208, 0.6)" },
      xaxis: "x",
      yaxis: "y2",
    };
  }

  function buildChartLayout() {
    return {
      grid: { rows: 2, columns: 1, pattern: "coupled", roworder: "top to bottom" },
      xaxis: {
        rangeslider: { visible: false },
        domain: [0, 1],
        anchor: "y",
      },
      yaxis: {
        domain: [0.35, 1],
        anchor: "x",
        title: { text: "价格", standoff: 10 },
      },
      yaxis2: {
        domain: [0, 0.28],
        anchor: "x",
        title: { text: "成交量" },
      },
      hovermode: "x unified",
      dragmode: "zoom",
      showlegend: true,
      legend: { orientation: "h", y: 1.12, x: 0, xanchor: "left" },
      margin: { t: 50, r: 40, b: 30, l: 50 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: { family: "'Segoe UI', 'Microsoft YaHei', sans-serif", color: "#202124" },
    };
  }

  function buildChartConfig() {
    return {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToAdd: ["drawrect", "eraseshape"],
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    };
  }

  function renderChart(chartData, displayPaths) {
    const card = getElement("chart-card");
    const root = getElement("chart");
    if (!root || typeof global.Plotly === "undefined") {
      return;
    }
    show(card);
    WB.clearChildren(root);

    const history = chartData?.history || [];
    const forecast = chartData?.forecast || {};
    if (history.length === 0 || !forecast.timestamps) {
      WB.setText(root, "图表数据不可用");
      return;
    }

    const historySeries = extractHistorySeries(history);
    const forecastDates = forecast.timestamps || [];
    const meanClose = forecast.mean_close || [];
    const q05 = forecast.q05_close || [];
    const q50 = forecast.q50_close || [];
    const q95 = forecast.q95_close || [];

    const traces = [];
    traces.push(buildCandlestickTrace(historySeries));
    traces.push.apply(traces, buildPathTraces(forecastDates, displayPaths));
    traces.push.apply(traces, buildQuantileTraces(forecastDates, q05, q50, q95));
    traces.push({
      x: forecastDates,
      y: meanClose,
      type: "scatter",
      mode: "lines",
      line: { color: "#0f2744", width: 2 },
      name: "均值路径",
      xaxis: "x",
      yaxis: "y",
    });
    traces.push(buildVolumeTrace(historySeries.dates, historySeries.volume));

    global.Plotly.newPlot(root, traces, buildChartLayout(), buildChartConfig());
  }

  function init() {
    const input = getElement("stock-code");
    const btn = getElement("generate-btn");
    const baselineBtn = getElement("baseline-btn");
    const suggestions = getElement("stock-suggestions");
    const retryBtn = getElement("error-retry-btn");

    renderRecent();
    loadStockList();

    if (input) {
      input.addEventListener("input", function () {
        updateSuggestions(input.value);
      });
      input.addEventListener("keydown", function (event) {
        if (event.key === "ArrowDown") {
          event.preventDefault();
          moveSuggestion(1);
        } else if (event.key === "ArrowUp") {
          event.preventDefault();
          moveSuggestion(-1);
        } else if (event.key === "Enter") {
          event.preventDefault();
          if (!applyActiveSuggestion()) {
            hide(suggestions);
            setInputExpanded(false);
            generateReport();
          }
        } else if (event.key === "Escape") {
          hide(suggestions);
          setInputExpanded(false);
          setActiveDescendant("");
        }
      });
      input.addEventListener("blur", function () {
        setTimeout(function () {
          hide(suggestions);
          setInputExpanded(false);
        }, 150);
      });
    }

    if (btn) {
      btn.addEventListener("click", function () {
        generateReport("model");
      });
    }
    if (baselineBtn) {
      baselineBtn.addEventListener("click", function () {
        generateReport("baseline");
      });
    }

    if (retryBtn) {
      retryBtn.addEventListener("click", function () {
        retryBtn.disabled = true;
        retryLastRequest();
      });
    }

    document.addEventListener("click", function (event) {
      if (suggestions && !suggestions.contains(event.target) && event.target !== input) {
        hide(suggestions);
        setInputExpanded(false);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})(window);
