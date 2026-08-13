/**
 * Kronos 组合页脚本。
 * 功能：本地组合资料的读取/编辑/保存、约束展示、模拟调仓。
 * 约束：全部 API 数据通过安全 DOM 方法渲染，不使用 innerHTML。
 */
(function (global) {
  "use strict";

  const WB = global.KronosWorkbench || {};

  const RISK_PROFILES = {
    conservative: { label: "稳健", stock: 0.05, sector: 0.20, cash: 0.20, turnover: 0.10, aversion: 8.0 },
    balanced: { label: "均衡", stock: 0.08, sector: 0.25, cash: 0.10, turnover: 0.20, aversion: 4.0 },
    aggressive: { label: "进取", stock: 0.12, sector: 0.30, cash: 0.05, turnover: 0.30, aversion: 2.0 },
  };

  const RISK_NAMES = {
    conservative: "稳健",
    balanced: "均衡",
    aggressive: "进取",
  };

  const CONSTRAINT_LABELS = [
    { key: "stock", name: "单股上限" },
    { key: "sector", name: "行业上限" },
    { key: "cash", name: "最低现金" },
    { key: "turnover", name: "单次换手" },
    { key: "aversion", name: "风险厌恶" },
  ];

  let serverPortfolio = null;
  let isDirty = false;
  let evaluationPassed = false;
  let isSaving = false;
  let isRebalancing = false;

  const elements = {};

  function cacheElements() {
    const ids = [
      "summary-cash", "summary-holdings", "summary-risk", "summary-version",
      "profile-name", "risk-profile", "cash",
      "holdings-table", "holdings-body",
      "constraints-grid", "sector-constraint-note",
      "gate-badge", "rebalance-btn", "rebalance-status", "rebalance-result",
      "target-weights-body", "orders-body", "rebalance-summary",
      "portfolio-action-bar", "action-bar-hint", "unsaved-badge",
      "save-btn", "discard-btn", "add-holding-btn",
    ];
    ids.forEach(function (id) {
      elements[id] = document.getElementById(id);
    });
  }

  function formatCurrency(value) {
    const num = Number(value);
    if (value == null || Number.isNaN(num)) {
      return "--";
    }
    return num.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function formatPercent(value) {
    return WB.formatPercent ? WB.formatPercent(value, 1) : String(value);
  }

  function setStatus(element, type, message) {
    WB.clearChildren(element);
    if (!message) {
      return;
    }
    const badge = WB.appendChild(element, "span", { className: "wb-badge wb-badge-" + type });
    WB.setText(badge, message);
  }

  function markDirty() {
    isDirty = true;
    if (elements["unsaved-badge"]) {
      elements["unsaved-badge"].classList.remove("hidden");
    }
    if (elements["action-bar-hint"]) {
      elements["action-bar-hint"].classList.remove("hidden");
    }
  }

  function markClean() {
    isDirty = false;
    if (elements["unsaved-badge"]) {
      elements["unsaved-badge"].classList.add("hidden");
    }
    if (elements["action-bar-hint"]) {
      elements["action-bar-hint"].classList.add("hidden");
    }
  }

  function beforeUnloadHandler(event) {
    if (!isDirty) {
      return;
    }
    event.preventDefault();
    event.returnValue = "有未保存的修改，确定要离开吗？";
  }

  function validateSixDigitCode(value) {
    return /^\d{6}$/.test(String(value || ""));
  }

  function validateNonNegativeInteger(value) {
    const str = String(value || "");
    return /^\d+$/.test(str) && Number(str) >= 0;
  }

  function validateNonNegativeNumber(value) {
    const num = Number(value);
    return !Number.isNaN(num) && num >= 0;
  }

  function setInputValid(input, valid) {
    if (valid) {
      input.classList.remove("is-invalid");
    } else {
      input.classList.add("is-invalid");
    }
  }

  function attachDirtyListeners(container) {
    if (!container) {
      return;
    }
    container.addEventListener("input", function (event) {
      if (event.target && event.target.classList.contains("cell-input")) {
        const input = event.target;
        if (input.classList.contains("cell-code")) {
          setInputValid(input, validateSixDigitCode(input.value));
        } else if (input.classList.contains("cell-shares")) {
          setInputValid(input, validateNonNegativeInteger(input.value));
        } else if (input.classList.contains("cell-cost")) {
          setInputValid(input, validateNonNegativeNumber(input.value));
        }
        markDirty();
      }
    });
  }

  function createHoldingInput(value, className, type, min, step) {
    const input = document.createElement("input");
    input.type = type || "text";
    input.className = "cell-input " + className;
    input.value = value == null ? "" : String(value);
    if (min != null) {
      input.min = String(min);
    }
    if (step != null) {
      input.step = String(step);
    }
    return input;
  }

  function createRemoveButton() {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "wb-btn wb-btn-secondary wb-btn-sm";
    btn.textContent = "删除";
    btn.addEventListener("click", function () {
      const row = btn.closest("tr");
      if (row) {
        row.remove();
        renderEmptyRowIfNeeded();
        markDirty();
      }
    });
    return btn;
  }

  function createHoldingRow(holding) {
    const row = document.createElement("tr");
    const codeCell = document.createElement("td");
    const codeInput = createHoldingInput(holding.stock_code, "cell-code", "text");
    codeCell.appendChild(codeInput);
    row.appendChild(codeCell);

    const sharesCell = document.createElement("td");
    const sharesInput = createHoldingInput(holding.shares, "cell-shares", "number", 0, 1);
    sharesCell.appendChild(sharesInput);
    row.appendChild(sharesCell);

    const costCell = document.createElement("td");
    const costInput = createHoldingInput(holding.cost_basis, "cell-cost", "number", 0, 0.01);
    costCell.appendChild(costInput);
    row.appendChild(costCell);

    const actionCell = document.createElement("td");
    actionCell.className = "col-action";
    actionCell.appendChild(createRemoveButton());
    row.appendChild(actionCell);

    return row;
  }

  function renderEmptyRowIfNeeded() {
    const body = elements["holdings-body"];
    if (!body) {
      return;
    }
    const hasRows = body.querySelectorAll("tr:not(.portfolio-empty-row)").length > 0;
    const emptyRow = body.querySelector(".portfolio-empty-row");
    if (hasRows && emptyRow) {
      emptyRow.remove();
    } else if (!hasRows && !emptyRow) {
      const row = document.createElement("tr");
      row.className = "portfolio-empty-row";
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.textContent = "暂无持仓，点击右上角添加。";
      row.appendChild(cell);
      body.appendChild(row);
    }
  }

  function renderHoldings(holdings) {
    const body = elements["holdings-body"];
    if (!body) {
      return;
    }
    WB.clearChildren(body);
    const list = Array.isArray(holdings) ? holdings : [];
    list.forEach(function (holding) {
      body.appendChild(createHoldingRow(holding));
    });
    renderEmptyRowIfNeeded();
  }

  function getSelectedRiskProfile() {
    const radios = document.querySelectorAll('input[name="risk-profile"]');
    for (let i = 0; i < radios.length; i++) {
      if (radios[i].checked) {
        return radios[i].value;
      }
    }
    return "balanced";
  }

  function setSelectedRiskProfile(value) {
    const radios = document.querySelectorAll('input[name="risk-profile"]');
    let matched = false;
    radios.forEach(function (radio) {
      const checked = radio.value === value;
      radio.checked = checked;
      if (checked) {
        matched = true;
      }
    });
    if (!matched && radios.length > 0) {
      radios[1].checked = true;
    }
  }

  function bindRiskProfileChange(handler) {
    const radios = document.querySelectorAll('input[name="risk-profile"]');
    radios.forEach(function (radio) {
      radio.addEventListener("change", handler);
    });
  }

  function renderConstraints(profileKey) {
    const grid = elements["constraints-grid"];
    if (!grid) {
      return;
    }
    WB.clearChildren(grid);
    const profile = RISK_PROFILES[profileKey] || RISK_PROFILES.balanced;
    CONSTRAINT_LABELS.forEach(function (item) {
      const card = WB.appendChild(grid, "div", { className: "constraint-card" });
      WB.appendChild(card, "div", { className: "constraint-name", textContent: item.name });
      const value = item.key === "aversion" ? profile[item.key] : formatPercent(profile[item.key]);
      WB.appendChild(card, "div", { className: "constraint-value", textContent: value });
    });
  }

  function renderSummary(portfolio) {
    WB.setText(elements["summary-cash"], formatCurrency(portfolio.cash));
    WB.setText(elements["summary-holdings"], String((portfolio.holdings || []).length));
    WB.setText(elements["summary-risk"], RISK_NAMES[portfolio.risk_profile] || portfolio.risk_profile || "--");
    WB.setText(elements["summary-version"], String(portfolio.version || 1));
  }

  function renderPortfolio(portfolio) {
    serverPortfolio = portfolio;
    renderSummary(portfolio);
    if (elements["profile-name"]) {
      elements["profile-name"].value = portfolio.name || "";
    }
    setSelectedRiskProfile(portfolio.risk_profile || "balanced");
    if (elements["cash"]) {
      elements["cash"].value = Number(portfolio.cash || 0).toFixed(2);
    }
    renderHoldings(portfolio.holdings);
    renderConstraints(portfolio.risk_profile || "balanced");
    markClean();
  }

  function collectHoldings() {
    const rows = elements["holdings-body"].querySelectorAll("tr:not(.portfolio-empty-row)");
    const holdings = [];
    let valid = true;
    rows.forEach(function (row) {
      const codeInput = row.querySelector(".cell-code");
      const sharesInput = row.querySelector(".cell-shares");
      const costInput = row.querySelector(".cell-cost");
      const code = codeInput.value.trim();
      const shares = sharesInput.value.trim();
      const cost = costInput.value.trim();

      const codeValid = validateSixDigitCode(code);
      const sharesValid = validateNonNegativeInteger(shares);
      const costValid = validateNonNegativeNumber(cost);

      setInputValid(codeInput, codeValid);
      setInputValid(sharesInput, sharesValid);
      setInputValid(costInput, costValid);

      if (!codeValid || !sharesValid || !costValid) {
        valid = false;
        return;
      }
      holdings.push({ stock_code: code, shares: parseInt(shares, 10), cost_basis: parseFloat(cost) });
    });
    return valid ? holdings : null;
  }

  function buildPortfolioPayload() {
    const name = elements["profile-name"].value.trim() || "默认组合";
    const riskProfile = getSelectedRiskProfile();
    const cash = parseFloat(elements["cash"].value || "0");
    const holdings = collectHoldings();
    if (holdings == null) {
      return null;
    }
    return { name: name, risk_profile: riskProfile, cash: cash, holdings: holdings };
  }

  function validatePayload(payload) {
    if (!RISK_PROFILES[payload.risk_profile]) {
      return "请选择有效的风险档案。";
    }
    if (Number.isNaN(payload.cash) || payload.cash < 0) {
      return "现金不能为负数。";
    }
    return null;
  }

  function updateRebalanceButton() {
    const btn = elements["rebalance-btn"];
    if (!btn) {
      return;
    }
    btn.disabled = !evaluationPassed || isDirty || isRebalancing;
  }

  function showApiError(container, error) {
    WB.clearChildren(container);
    const box = WB.appendChild(container, "div", { className: "wb-error" });
    WB.appendChild(box, "div", { className: "wb-error-title", textContent: error.code || "错误" });
    WB.appendChild(box, "div", { className: "error-message", textContent: error.message || "请求失败" });
    const details = error.details || {};
    if (Array.isArray(details.reason_codes) && details.reason_codes.length > 0) {
      WB.appendChild(box, "div", { className: "error-details", textContent: "原因：" + details.reason_codes.join("、") });
    }
  }

  function renderRecoveryTips(container, failedCodes) {
    const codes = Array.isArray(failedCodes) ? failedCodes : [];
    if (codes.length === 0) {
      return;
    }
    const tips = WB.appendChild(container, "div", { className: "wb-error" });
    WB.appendChild(tips, "div", { className: "wb-error-title", textContent: "门禁未通过" });
    const list = WB.appendChild(tips, "ul", { className: "error-details" });
    codes.forEach(function (code) {
      WB.appendChild(list, "li", { textContent: code });
    });
    WB.appendChild(tips, "div", { className: "error-details", textContent: "恢复提示：完成新的正式评估运行并确保 gate_result.passed 为 true 后刷新页面。" });
  }

  function renderTargetWeights(weights) {
    const body = elements["target-weights-body"];
    if (!body) {
      return;
    }
    WB.clearChildren(body);
    const entries = Object.entries(weights || {});
    if (entries.length === 0) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 2;
      cell.textContent = "无目标权重";
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    entries.sort(function (a, b) {
      return b[1] - a[1];
    });
    entries.forEach(function (entry) {
      const row = document.createElement("tr");
      const codeCell = document.createElement("td");
      codeCell.textContent = entry[0];
      row.appendChild(codeCell);
      const weightCell = document.createElement("td");
      weightCell.textContent = formatPercent(entry[1]);
      row.appendChild(weightCell);
      body.appendChild(row);
    });
  }

  function renderOrders(orders) {
    const body = elements["orders-body"];
    if (!body) {
      return;
    }
    WB.clearChildren(body);
    const list = Array.isArray(orders) ? orders : [];
    if (list.length === 0) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 6;
      cell.textContent = "无订单";
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    list.forEach(function (order) {
      const row = document.createElement("tr");
      const values = [
        order.stock_code,
        order.side,
        String(order.shares),
        WB.formatNumber ? WB.formatNumber(order.reference_price) : String(order.reference_price),
        formatCurrency(order.notional),
        formatCurrency(order.estimated_fee),
      ];
      values.forEach(function (value) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  function renderRebalanceSummary(rebalance) {
    const container = elements["rebalance-summary"];
    if (!container) {
      return;
    }
    WB.clearChildren(container);
    const items = [
      { key: "调仓方法", value: rebalance.method || "--" },
      { key: "现金权重", value: formatPercent(rebalance.cash_weight) },
      { key: "剩余现金", value: formatCurrency(rebalance.remaining_cash) },
      { key: "估算总费用", value: formatCurrency(rebalance.estimated_fees) },
    ];
    items.forEach(function (item) {
      const row = WB.appendChild(container, "div", { className: "kv-item" });
      WB.appendChild(row, "span", { className: "kv-key", textContent: item.key });
      WB.appendChild(row, "span", { className: "kv-value", textContent: item.value });
    });
  }

  function showRebalanceResult(rebalance) {
    const resultBox = elements["rebalance-result"];
    if (!resultBox) {
      return;
    }
    resultBox.classList.remove("hidden");
    renderTargetWeights(rebalance.target_weights);
    renderOrders(rebalance.orders);
    renderRebalanceSummary(rebalance);
  }

  function hideRebalanceResult() {
    const resultBox = elements["rebalance-result"];
    if (resultBox) {
      resultBox.classList.add("hidden");
    }
  }

  async function loadPortfolio() {
    const result = await WB.getJson("/api/v2/portfolio");
    if (!result.ok) {
      showApiError(elements["rebalance-status"], result.error);
      return;
    }
    renderPortfolio(result.payload.portfolio);
    updateRebalanceButton();
  }

  async function savePortfolio() {
    if (isSaving) {
      return;
    }
    const payload = buildPortfolioPayload();
    if (payload == null) {
      setStatus(elements["rebalance-status"], "error", "持仓数据不合法，请检查输入。");
      return;
    }
    const validationError = validatePayload(payload);
    if (validationError) {
      setStatus(elements["rebalance-status"], "error", validationError);
      return;
    }

    isSaving = true;
    elements["save-btn"].disabled = true;
    setStatus(elements["rebalance-status"], "warn", "保存中…");

    const result = await WB.putJson("/api/v2/portfolio", payload);
    elements["save-btn"].disabled = false;
    isSaving = false;

    if (!result.ok) {
      showApiError(elements["rebalance-status"], result.error);
      return;
    }

    setStatus(elements["rebalance-status"], "ok", "已保存（本地）");
    await loadPortfolio();
    updateRebalanceButton();
  }

  async function discardChanges() {
    await loadPortfolio();
    setStatus(elements["rebalance-status"], "info", "已恢复为服务器保存的版本。");
    hideRebalanceResult();
    updateRebalanceButton();
  }

  async function loadEvaluation() {
    const badge = elements["gate-badge"];
    const result = await WB.getJson("/api/v2/evaluation/latest");
    if (!result.ok) {
      evaluationPassed = false;
      WB.setText(badge, "评估门禁不可用");
      badge.className = "wb-badge wb-badge-error";
      if (result.status === 404) {
        renderRecoveryTips(elements["rebalance-status"], ["EVALUATION_NOT_FOUND"]);
      } else {
        showApiError(elements["rebalance-status"], result.error);
      }
      updateRebalanceButton();
      return;
    }
    const gate = result.payload.gate_result || {};
    if (gate.passed) {
      evaluationPassed = true;
      WB.setText(badge, "评估门禁通过");
      badge.className = "wb-badge wb-badge-ok";
    } else {
      evaluationPassed = false;
      WB.setText(badge, "评估门禁未通过");
      badge.className = "wb-badge wb-badge-error";
      renderRecoveryTips(elements["rebalance-status"], gate.failed_codes || []);
    }
    updateRebalanceButton();
  }

  async function runRebalance() {
    if (isRebalancing || !evaluationPassed || isDirty) {
      return;
    }
    isRebalancing = true;
    elements["rebalance-btn"].disabled = true;
    setStatus(elements["rebalance-status"], "warn", "生成中…");
    hideRebalanceResult();

    const result = await WB.postJson("/api/v2/portfolio/rebalance", { portfolio_id: "default" });

    isRebalancing = false;
    updateRebalanceButton();

    if (!result.ok) {
      showApiError(elements["rebalance-status"], result.error);
      return;
    }

    setStatus(elements["rebalance-status"], "ok", "模拟调仓已生成");
    showRebalanceResult(result.payload.rebalance);
  }

  function addHoldingRow() {
    const body = elements["holdings-body"];
    if (!body) {
      return;
    }
    const emptyRow = body.querySelector(".portfolio-empty-row");
    if (emptyRow) {
      emptyRow.remove();
    }
    body.appendChild(createHoldingRow({ stock_code: "", shares: 0, cost_basis: 0 }));
    markDirty();
  }

  function bindEvents() {
    elements["save-btn"].addEventListener("click", savePortfolio);
    elements["discard-btn"].addEventListener("click", discardChanges);
    elements["add-holding-btn"].addEventListener("click", addHoldingRow);
    elements["rebalance-btn"].addEventListener("click", runRebalance);

    elements["profile-name"].addEventListener("input", markDirty);
    elements["cash"].addEventListener("input", function () {
      setInputValid(elements["cash"], validateNonNegativeNumber(elements["cash"].value));
      markDirty();
    });
    bindRiskProfileChange(function () {
      renderConstraints(getSelectedRiskProfile());
      markDirty();
    });

    attachDirtyListeners(elements["holdings-body"]);
    window.addEventListener("beforeunload", beforeUnloadHandler);
  }

  function init() {
    cacheElements();
    bindEvents();
    loadPortfolio();
    loadEvaluation();
  }

  document.addEventListener("DOMContentLoaded", init);
})(window);
