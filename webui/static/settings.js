/**
 * Kronos 系统诊断页：只读展示配置与最新证据，不写回配置。
 * 约束：仅调用 GET /api/config 与 GET /api/v2/evaluation/latest；不使用输入控件。
 */
(function (global) {
  "use strict";

  var WB = global.KronosWorkbench;

  var MODEL_FIELDS = [
    ["tokenizer", "Tokenizer"],
    ["predictor", "模型"],
    ["tokenizer_revision", "Tokenizer revision"],
    ["model_revision", "模型 revision"],
    ["max_context", "最大上下文"],
    ["device", "设备"],
    ["idle_timeout_minutes", "空闲超时（分钟）"]
  ];

  var PREDICTION_FIELDS = [
    ["default_pred_len", "默认预测期限"],
    ["default_sample_count", "默认采样数量"],
    ["default_temperature", "温度 T"],
    ["default_top_p", "Top-P"],
    ["timeout_seconds", "推理超时（秒）"],
    ["fallback_sample_count", "降级采样数量"]
  ];

  var DATA_FIELDS = [
    ["primary_source", "主源"],
    ["backup_source", "备源"],
    ["optional_sources", "可选源"],
    ["cache_dir", "缓存目录"],
    ["cache_ttl_hours", "缓存有效期（小时）"],
    ["retry_max", "最大重试"],
    ["retry_backoff_seconds", "退避间隔（秒）"]
  ];

  var WEBUI_FIELDS = [
    ["host", "监听地址"],
    ["port", "端口"],
    ["stock_pool", "默认股票池"]
  ];

  var EVIDENCE_HASH_FIELDS = [
    ["model_hash", "model_hash"],
    ["rules_hash", "rules_hash"],
    ["config_hash", "config_hash"],
    ["data_hash", "data_hash"]
  ];

  function init() {
    bindCopyButtons();
    loadConfig();
    loadEvidence();
  }

  function bindCopyButtons() {
    document.body.addEventListener("click", function (event) {
      var button = event.target.closest(".wb-copy-btn");
      if (!button) {
        return;
      }
      var targetId = button.getAttribute("data-target");
      if (targetId) {
        copyTargetText(targetId, button);
      }
    });
  }

  function copyTargetText(targetId, button) {
    var source = document.getElementById(targetId);
    var text = source ? source.textContent : "";
    if (!text) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        showCopyFeedback(button, "已复制");
      }, function () {
        showCopyFeedback(button, "复制失败，请手动复制");
      });
    } else {
      showCopyFeedback(button, "浏览器不支持自动复制");
    }
  }

  function showCopyFeedback(button, message) {
    var previous = button.textContent;
    WB.setText(button, message);
    setTimeout(function () {
      WB.setText(button, previous);
    }, 1500);
  }

  async function loadConfig() {
    var result = await WB.getJson("/api/config");
    if (!result.ok) {
      renderError("model-dl", "配置读取失败：" + errorMessage(result));
      return;
    }
    renderConfig(result.payload);
  }

  function renderConfig(config) {
    renderDefinitionList("model-dl", config.model, MODEL_FIELDS);
    renderDefinitionList("prediction-dl", config.prediction, PREDICTION_FIELDS);
    renderDefinitionList("data-dl", config.data, DATA_FIELDS);
    renderDefinitionList("webui-dl", config.webui, WEBUI_FIELDS);
  }

  function renderDefinitionList(dlId, section, fields) {
    var dl = document.getElementById(dlId);
    if (!dl) return;
    WB.clearChildren(dl);
    if (!section || typeof section !== "object") return;
    fields.forEach(function (pair) {
      appendDefinition(dl, pair[1], formatValue(section[pair[0]]));
    });
  }

  async function loadEvidence() {
    var result = await WB.getJson("/api/v2/evaluation/latest");
    var dl = document.getElementById("evidence-dl");
    if (!dl) return;
    WB.clearChildren(dl);
    if (!result.ok) {
      WB.appendChild(dl, "div", {
        className: "wb-note",
        textContent: result.status === 404
          ? "尚无正式评估产物。运行 python -m evaluation.run --stage smoke --as-of YYYY-MM-DD。"
          : "证据读取失败：" + errorMessage(result)
      });
      return;
    }
    renderEvidence(result.payload, dl);
  }

  function renderEvidence(payload, dl) {
    var manifest = payload.manifest || {};
    var gate = payload.gate_result || {};
    var checks = gate.checks || {};

    appendDefinition(dl, "最近运行", payload.run_id);
    appendDefinition(dl, "状态", manifest.status);
    appendDefinition(dl, "门禁", renderGateSummary(gate));
    appendDefinition(dl, "版本匹配", renderVersionMatch(checks.VERSION_MATCH));
    appendDefinition(dl, "模型", manifest.model);

    EVIDENCE_HASH_FIELDS.forEach(function (pair) {
      var key = pair[0];
      var label = pair[1];
      var value = manifest[key];
      var targetId = "evidence-" + key;
      appendDefinitionWithCopy(dl, label, value, targetId);
    });

    WB.appendChild(dl, "div", {
      className: "wb-note",
      textContent: "评估产物 30 天内有效；外部修改配置后需重启，且哈希可能不再匹配。"
    });
  }

  function renderGateSummary(gate) {
    if (!gate || typeof gate !== "object" || !gate.checks) {
      return "无门禁结果";
    }
    if (gate.passed) {
      return "通过";
    }
    return "未通过";
  }

  function renderVersionMatch(value) {
    if (value === true) {
      return "匹配";
    }
    if (value === false) {
      return "不匹配";
    }
    return "无结果";
  }

  function appendDefinition(dl, term, value) {
    WB.appendChild(dl, "dt", { textContent: term });
    WB.appendChild(dl, "dd", { textContent: value == null ? "--" : value });
  }

  function appendDefinitionWithCopy(dl, term, value, targetId) {
    WB.appendChild(dl, "dt", { textContent: term });
    var dd = document.createElement("dd");
    dd.className = "hash-line";
    WB.appendChild(dd, "code", { id: targetId, textContent: value == null ? "--" : value });
    if (value) {
      WB.appendChild(dd, "button", {
        type: "button",
        className: "wb-copy-btn wb-btn-sm",
        "data-target": targetId,
        ariaLabel: "复制 " + term,
        textContent: "复制"
      });
    }
    dl.appendChild(dd);
  }

  function formatValue(value) {
    if (value == null) return "--";
    if (Array.isArray(value)) return value.length ? value.join(", ") : "--";
    if (typeof value === "boolean") return value ? "是" : "否";
    if (typeof value === "number" && !Number.isFinite(value)) return "--";
    return String(value);
  }

  function errorMessage(result) {
    return result.error && result.error.message ? result.error.message : "未知错误";
  }

  function renderError(dlId, message) {
    var dl = document.getElementById(dlId);
    if (!dl) return;
    WB.clearChildren(dl);
    WB.appendChild(dl, "div", { className: "wb-note", textContent: message });
  }

  document.addEventListener("DOMContentLoaded", init);
})(window);
