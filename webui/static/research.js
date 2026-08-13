/**
 * Kronos 研究证据页：只读展示最新评估运行与前瞻台账。
 * 约束：安全 DOM，不 innerHTML；仅调用 GET /api/v2/evaluation/latest 与 /api/v2/forward-ledger/latest。
 */
(function (global) {
  "use strict";

  var WB = global.KronosWorkbench;
  var HORIZONS = [5, 20, 60];

  var EMPTY_SMOKE_COMMAND = "python -m evaluation.run --stage smoke --as-of YYYY-MM-DD";
  var EMPTY_FULL_COMMAND = "python -m evaluation.run --stage full --as-of YYYY-MM-DD";

  var GATE_LABELS = {
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
    VERSION_MATCH: "版本匹配",
    FORMAL_PROTOCOL: "完整正式研究协议（含沪深300全收益基准）",
    FRESH_ARTIFACT: "评估产物30天内"
  };

  var METRIC_LABELS = {
    rank_ic_mean: "样本外 RankIC 均值",
    coverage_80: "80%区间覆盖率",
    up_probability_ece: "上涨概率 ECE",
    annualized_excess_return: "年化超额收益",
    information_ratio: "信息比率",
    drawdown_worsening: "回撤恶化",
    positive_12m_window_ratio: "正超额12月窗口比例"
  };

  var METRIC_DIGITS = {
    rank_ic_mean: 4,
    coverage_80: 1,
    up_probability_ece: 4,
    annualized_excess_return: 2,
    information_ratio: 2,
    drawdown_worsening: 2,
    positive_12m_window_ratio: 1
  };

  var PERCENT_METRICS = {
    coverage_80: true,
    positive_12m_window_ratio: true,
    annualized_excess_return: true,
    drawdown_worsening: true
  };

  function init() {
    bindCopyButtons();
    loadResearch();
    loadDailyForward();
  }

  function bindCopyButtons() {
    document.querySelectorAll(".wb-copy-btn").forEach(function (button) {
      button.addEventListener("click", function () {
        copyCommand(button.getAttribute("data-target"));
      });
    });
  }

  function copyCommand(targetId) {
    var container = document.getElementById(targetId);
    if (!container) return;
    var code = container.querySelector("code");
    var text = code ? code.textContent : "";
    if (!text) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        showCopyFeedback(container, "已复制");
      }, function () {
        showCopyFeedback(container, "复制失败，请手动复制");
      });
    } else {
      showCopyFeedback(container, "浏览器不支持自动复制，请手动复制");
    }
  }

  function showCopyFeedback(container, message) {
    var button = container.querySelector(".wb-copy-btn");
    if (!button) return;
    var previous = button.textContent;
    WB.setText(button, message);
    setTimeout(function () {
      WB.setText(button, previous);
    }, 1500);
  }

  async function loadResearch() {
    var result = await WB.getJson("/api/v2/evaluation/latest");
    if (!result.ok) {
      if (result.status === 404) {
        renderEmpty();
        return;
      }
      renderError("评估读取失败：" + (result.error && result.error.message ? result.error.message : "未知错误"));
      return;
    }
    renderRun(result.payload);
  }

  async function loadDailyForward() {
    var result = await WB.getJson("/api/v2/forward-ledger/latest");
    var card = document.getElementById("daily-forward-card");
    var body = document.getElementById("daily-forward-body");
    if (!card || !body) return;
    card.hidden = false;
    WB.clearChildren(body);
    if (!result.ok) {
      WB.setText(body, result.status === 404
        ? "尚未生成每日建议前瞻台账。每日收盘后保存建议，满20个交易日后再运行本地台账评估。"
        : "前瞻台账读取失败：" + (result.error && result.error.message ? result.error.message : "未知错误"));
      return;
    }
    renderDailyForward(result.payload, body);
  }

  function renderRun(payload) {
    var manifest = payload.manifest || {};
    renderStatus(manifest, payload.run_id);
    renderGate(payload.gate_result);
    renderMetrics(manifest.metrics);
    renderCalibration(getCalibrationMetrics(manifest));
    renderBaselines(manifest.baselines);
    renderForward(manifest.forward_validation);
    renderDetails(manifest, payload.run_id);
  }

  function renderEmpty() {
    showCard("empty-card", true);
  }

  function renderError(message) {
    var card = document.getElementById("empty-card");
    var title = card ? card.querySelector(".wb-empty-title") : null;
    if (title) WB.setText(title, message);
    showCard("empty-card", true);
  }

  function renderStatus(manifest, runId) {
    var phase = derivePhase(manifest);
    var badge = document.getElementById("run-badge");
    var label = document.getElementById("run-phase");
    if (badge) {
      badge.className = "wb-badge " + phase.cls;
      WB.setText(badge, phase.label);
    }
    if (label) WB.setText(label, manifest.status || "--");

    var meta = document.getElementById("status-meta");
    if (!meta) return;
    WB.clearChildren(meta);
    appendStatusItem(meta, "run_id", runId);
    appendStatusItem(meta, "截止日", manifest.as_of);
    appendStatusItem(meta, "新鲜度", freshness(manifest.completed_at || manifest.started_at));
    appendStatusItem(meta, "失败原因", formatFailureReason(manifest.failure_reason));
  }

  function derivePhase(manifest) {
    // collection_only / blocked / diagnostic 工件不得误标为正式
    if (manifest.collection_only || manifest.blocked || manifest.diagnostic) {
      return { label: "收集/诊断", cls: "wb-badge-warn" };
    }
    if (manifest.formal_protocol) return { label: "正式", cls: "wb-badge-ok" };
    var command = manifest.command || "";
    if (command.indexOf("--stage smoke") >= 0 || manifest.status === "completed-smoke") {
      return { label: "Smoke", cls: "wb-badge-info" };
    }
    return { label: manifest.status || "未知", cls: "wb-badge-info" };
  }

  function appendStatusItem(parent, label, value) {
    var item = document.createElement("div");
    item.className = "status-item";
    WB.appendChild(item, "span", { className: "status-item-label", textContent: label });
    WB.appendChild(item, "span", { className: "status-item-value", textContent: value == null ? "--" : value });
    parent.appendChild(item);
  }

  function freshness(iso) {
    if (!iso) return "--";
    var then = new Date(iso);
    if (Number.isNaN(then.getTime())) return String(iso);
    var days = Math.floor((Date.now() - then.getTime()) / (1000 * 60 * 60 * 24));
    if (days < 0) return "未来";
    if (days === 0) return "今天";
    if (days <= 30) return days + " 天前";
    return "已过期（" + days + " 天前）";
  }

  function formatFailureReason(reasons) {
    if (!Array.isArray(reasons) || !reasons.length) return "无";
    return reasons.join("；");
  }

  function createIcon(useHref, className) {
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", className);
    svg.setAttribute("aria-hidden", "true");
    var use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", useHref);
    svg.appendChild(use);
    return svg;
  }

  function renderGate(gate) {
    var card = document.getElementById("gate-card");
    var summary = document.getElementById("gate-summary");
    var list = document.getElementById("gate-list");
    if (!card || !summary || !list) return;
    card.hidden = false;
    WB.clearChildren(summary);
    WB.clearChildren(list);

    if (!gate || typeof gate !== "object" || !gate.checks) {
      WB.appendChild(summary, "span", { className: "gate-status gate-fail", textContent: "无门禁结果" });
      WB.appendChild(list, "div", { className: "gate-empty", textContent: "尚无门禁检查数据。" });
      return;
    }

    WB.appendChild(summary, "span", {
      className: gate.passed ? "gate-pass" : "gate-fail",
      textContent: gate.passed ? "门禁通过" : "门禁未通过"
    });

    sortGateEntries(gate.checks, gate.failed_codes).forEach(function (entry) {
      appendGateRow(list, entry.code, entry.passed);
    });
  }

  function sortGateEntries(checks, failedCodes) {
    var failed = new Set(Array.isArray(failedCodes) ? failedCodes : []);
    var entries = Object.keys(checks).map(function (code) {
      return { code: code, passed: Boolean(checks[code]) };
    });
    entries.sort(function (a, b) {
      var af = failed.has(a.code) ? 0 : 1;
      var bf = failed.has(b.code) ? 0 : 1;
      if (af !== bf) return af - bf;
      return a.code.localeCompare(b.code);
    });
    return entries;
  }

  function appendGateRow(list, code, passed) {
    var row = document.createElement("div");
    row.className = "gate-row";
    WB.appendChild(row, "span", { className: "gate-label", textContent: GATE_LABELS[code] || code });
    var status = WB.appendChild(row, "span", {
      className: passed ? "gate-status gate-pass" : "gate-status gate-fail",
    });
    var iconHref = passed ? "#icon-check" : "#icon-x";
    status.appendChild(createIcon(iconHref, "icon gate-icon"));
    status.appendChild(WB.textNode(passed ? "通过" : "未通过"));
    list.appendChild(row);
  }

  function renderMetrics(metrics) {
    var card = document.getElementById("metrics-card");
    var grid = document.getElementById("metrics-grid");
    if (!card || !grid) return;
    if (!metrics || typeof metrics !== "object") {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    WB.clearChildren(grid);
    Object.keys(METRIC_LABELS).forEach(function (key) {
      var value = metrics[key];
      appendMetricCard(grid, METRIC_LABELS[key], formatMetricValue(key, value));
    });
  }

  function formatMetricValue(key, value) {
    if (value == null) return "--";
    if (Number.isNaN(Number(value))) return "--";
    if (PERCENT_METRICS[key]) {
      return WB.formatPercent(value, METRIC_DIGITS[key]);
    }
    return WB.formatNumber(value, METRIC_DIGITS[key]);
  }

  function appendMetricCard(grid, label, value) {
    var card = document.createElement("div");
    card.className = "metric-card";
    WB.appendChild(card, "div", { className: "metric-card-label", textContent: label });
    WB.appendChild(card, "div", { className: "metric-card-value", textContent: value });
    grid.appendChild(card);
  }

  function getCalibrationMetrics(manifest) {
    var metrics = manifest.metrics || {};
    if (metrics.calibration_by_horizon && typeof metrics.calibration_by_horizon === "object") {
      return metrics.calibration_by_horizon;
    }
    if (metrics.calibration && metrics.calibration.metrics) {
      return metrics.calibration.metrics;
    }
    return null;
  }

  function renderCalibration(byHorizon) {
    var card = document.getElementById("calibration-card");
    var grid = document.getElementById("calibration-grid");
    if (!card || !grid) return;
    if (!byHorizon || typeof byHorizon !== "object") {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    WB.clearChildren(grid);
    HORIZONS.forEach(function (horizon) {
      var metric = byHorizon[String(horizon)];
      appendCalibrationCard(grid, horizon, metric);
    });
  }

  function appendCalibrationCard(grid, horizon, metric) {
    var card = document.createElement("div");
    card.className = "horizon-card";
    WB.appendChild(card, "div", { className: "horizon-title", textContent: horizon + "日校准" });
    var dl = createDefinitionList();
    appendDefinition(dl, "可用", metric && metric.available ? "是" : "否");
    appendDefinition(dl, "样本外锚点", safeInt(metric && metric.oos_anchor_count));
    appendDefinition(dl, "观测数", safeInt(metric && metric.oos_observation_count));
    appendDefinition(dl, "覆盖率", formatPercentOrNull(metric && metric.coverage_80));
    appendDefinition(dl, "ECE", formatNumberOrNull(metric && metric.up_probability_ece, 4));
    card.appendChild(dl);
    grid.appendChild(card);
  }

  function renderBaselines(baselines) {
    renderObjectCard("baselines-card", "baselines-body", baselines, "基线比较");
  }

  function renderForward(forward) {
    renderObjectCard("forward-card", "forward-body", forward, "前瞻验证");
  }

  function renderObjectCard(cardId, bodyId, data, fallbackTitle) {
    var card = document.getElementById(cardId);
    var body = document.getElementById(bodyId);
    if (!card || !body) return;
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    WB.clearChildren(body);
    var dl = createDefinitionList();
    Object.keys(data).forEach(function (key) {
      appendDefinition(dl, key, formatNestedValue(data[key]));
    });
    if (!dl.children.length) {
      WB.setText(body, "暂无数据。");
      return;
    }
    body.appendChild(dl);
  }

  function formatNestedValue(value) {
    if (value == null) return "--";
    if (typeof value === "number") return Number.isFinite(value) ? WB.formatNumber(value, 4) : "--";
    if (typeof value === "boolean") return value ? "是" : "否";
    if (typeof value === "object") {
      try {
        return JSON.stringify(value);
      } catch (e) {
        return String(value);
      }
    }
    return String(value);
  }

  function renderDetails(manifest, runId) {
    var card = document.getElementById("details-card");
    var body = document.getElementById("details-body");
    if (!card || !body) return;
    card.hidden = false;
    WB.clearChildren(body);
    var dl = createDefinitionList();
    appendDefinition(dl, "run_id", runId);
    appendDefinition(dl, "命令", manifest.command);
    appendDefinition(dl, "截止日", manifest.as_of);
    appendDefinition(dl, "锚点频率", manifest.anchor_frequency);
    appendDefinition(dl, "股票", formatArray(manifest.stocks));
    appendDefinition(dl, "模型", manifest.model);
    appendDefinition(dl, "Tokenizer", manifest.tokenizer);
    appendDefinition(dl, "Git 提交", manifest.git_commit);
    appendDefinition(dl, "工作区", manifest.worktree_dirty ? "脏" : "干净");
    appendDefinition(dl, "model_hash", manifest.model_hash);
    appendDefinition(dl, "config_hash", manifest.config_hash);
    appendDefinition(dl, "data_hash", manifest.data_hash);
    appendDefinition(dl, "rules_hash", manifest.rules_hash);
    appendDefinition(dl, "开始时间", WB.formatDate(manifest.started_at));
    appendDefinition(dl, "完成时间", WB.formatDate(manifest.completed_at));
    body.appendChild(dl);
  }

  function renderDailyForward(payload, body) {
    var report = payload.report || {};
    var dl = createDefinitionList();
    appendDefinition(dl, "工件", payload.artifact);
    appendDefinition(dl, "状态", report.status);
    appendDefinition(dl, "累计建议", safeInt(report.total_recommendations));
    appendDefinition(dl, "已成熟", safeInt(report.matured_recommendations));
    appendDefinition(dl, "待成熟", safeInt(report.pending_recommendations));
    appendDefinition(dl, "排除", safeInt(report.excluded_recommendations));
    Object.keys(report.by_action || {}).forEach(function (action) {
      var group = report.by_action[action];
      appendDefinition(dl, action + " 样本", safeInt(group.count));
      appendDefinition(dl, action + " 平均收益", WB.formatPercent(group.mean_actual_return, 2));
      appendDefinition(dl, action + " 上涨比例", WB.formatPercent(group.up_rate, 1));
    });
    body.appendChild(dl);
    if (report.disclaimer) {
      WB.appendChild(body, "p", { className: "wb-note", textContent: report.disclaimer });
    }
  }

  function createDefinitionList() {
    return document.createElement("dl");
  }

  function appendDefinition(dl, term, value) {
    var dt = document.createElement("dt");
    dt.textContent = term;
    var dd = document.createElement("dd");
    dd.textContent = value == null ? "--" : value;
    dl.appendChild(dt);
    dl.appendChild(dd);
  }

  function formatPercentOrNull(value) {
    return value == null || Number.isNaN(Number(value)) ? "--" : WB.formatPercent(value, 1);
  }

  function formatNumberOrNull(value, digits) {
    return value == null || Number.isNaN(Number(value)) ? "--" : WB.formatNumber(value, digits);
  }

  function safeInt(value) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return String(Math.floor(Number(value)));
  }

  function formatArray(value) {
    if (!Array.isArray(value)) return "--";
    if (!value.length) return "--";
    return value.join(", ");
  }

  function showCard(id, visible) {
    var el = document.getElementById(id);
    if (el) el.hidden = !visible;
  }

  document.addEventListener("DOMContentLoaded", init);
})(window);
