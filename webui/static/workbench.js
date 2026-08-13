/**
 * Kronos v2 工作台通用脚本。
 * 功能：侧栏折叠持久化、安全 DOM 辅助、通用 fetch 错误解析。
 * 约束：不依赖 innerHTML 渲染 API 数据。
 */
(function (global) {
  "use strict";

  const STORAGE_KEY = "kronos_workbench_sidebar";

  /**
   * 初始化侧栏折叠状态。
   * @param {string} sidebarId
   * @param {string} toggleId
   */
  function readStorage(key, fallback) {
    try {
      return global.localStorage.getItem(key);
    } catch (err) {
      return fallback;
    }
  }

  function writeStorage(key, value) {
    try {
      global.localStorage.setItem(key, value);
    } catch (err) {
      // 禁用 localStorage 时静默失败，侧栏仍可折叠
    }
  }

  function initSidebar(sidebarId, toggleId) {
    const sidebar = document.getElementById(sidebarId);
    const toggle = document.getElementById(toggleId);
    if (!sidebar || !toggle) {
      return;
    }

    const saved = readStorage(STORAGE_KEY, null);
    const collapsed = saved === "collapsed";
    if (collapsed) {
      sidebar.classList.add("collapsed");
      toggle.setAttribute("aria-expanded", "false");
    }

    toggle.addEventListener("click", function () {
      const isCollapsed = sidebar.classList.toggle("collapsed");
      toggle.setAttribute("aria-expanded", String(!isCollapsed));
      writeStorage(STORAGE_KEY, isCollapsed ? "collapsed" : "expanded");
    });
  }

  /**
   * 安全创建文本节点。
   * @param {string} text
   * @returns {Text}
   */
  function textNode(text) {
    return document.createTextNode(String(text == null ? "" : text));
  }

  /**
   * 安全设置元素文本内容。
   * @param {HTMLElement} element
   * @param {string} text
   */
  function setText(element, text) {
    if (!element) {
      return;
    }
    element.textContent = String(text == null ? "" : text);
  }

  /**
   * 清空子元素。
   * @param {HTMLElement} element
   */
  function clearChildren(element) {
    if (!element) {
      return;
    }
    while (element.firstChild) {
      element.removeChild(element.firstChild);
    }
  }

  /**
   * 创建子元素并追加到父元素。
   * @param {HTMLElement} parent
   * @param {string} tag
   * @param {Object} [attrs]
   * @param {Array<Node|string>} [children]
   * @returns {HTMLElement}
   */
  function appendChild(parent, tag, attrs, children) {
    const el = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(function (_ref) {
        const key = _ref[0];
        const value = _ref[1];
        if (key === "className") {
          el.className = String(value);
        } else if (key === "textContent") {
          setText(el, value);
        } else if (key.startsWith("data-")) {
          el.setAttribute(key, String(value));
        } else if (key === "ariaLabel") {
          el.setAttribute("aria-label", String(value));
        } else if (key === "ariaHidden") {
          el.setAttribute("aria-hidden", String(value));
        } else {
          el.setAttribute(key, String(value));
        }
      });
    }
    if (children) {
      children.forEach(function (child) {
        if (child == null) {
          return;
        }
        el.appendChild(typeof child === "string" || typeof child === "number" ? textNode(child) : child);
      });
    }
    if (parent) {
      parent.appendChild(el);
    }
    return el;
  }

  /**
   * 格式化百分比。
   * @param {number|null|undefined} value
   * @param {number} [digits]
   * @returns {string}
   */
  function formatPercent(value, digits) {
    digits = digits == null ? 1 : digits;
    if (value == null || Number.isNaN(Number(value))) {
      return "--";
    }
    return (Number(value) * 100).toFixed(digits) + "%";
  }

  /**
   * 格式化数字。
   * @param {number|null|undefined} value
   * @param {number} [digits]
   * @returns {string}
   */
  function formatNumber(value, digits) {
    digits = digits == null ? 2 : digits;
    if (value == null || Number.isNaN(Number(value))) {
      return "--";
    }
    return Number(value).toFixed(digits);
  }

  /**
   * 解析 fetch 错误为统一结构。
   * @param {Response|null} response
   * @param {Object} [payload]
   * @param {Error} [networkError]
   * @returns {{code: string, message: string, retryable: boolean, details: Object}}
   */
  function parseApiError(response, payload, networkError) {
    if (networkError) {
      return {
        code: "NETWORK_ERROR",
        message: "网络请求失败：" + networkError.message,
        retryable: true,
        details: {},
      };
    }
    if (!response) {
      return {
        code: "UNKNOWN_ERROR",
        message: "请求失败，未收到响应。",
        retryable: true,
        details: {},
      };
    }
    const body = payload && typeof payload === "object" ? payload : {};
    const error = body.error && typeof body.error === "object" ? body.error : {};
    return {
      code: error.code || "HTTP_" + response.status,
      message: error.message || "HTTP " + response.status,
      retryable: Boolean(error.retryable),
      details: error.details || {},
    };
  }

  /**
   * 带超时与 JSON 解析的通用请求。
   * @param {string} url
   * @param {{method?: string, body?: Object, timeoutMs?: number}} [options]
   * @returns {Promise<{ok: boolean, status: number, payload: Object, error: Object|null}>}
   */
  async function requestJson(url, options) {
    options = options || {};
    const method = options.method || "GET";
    const hasBody = options.body !== undefined;
    const timeoutMs = options.timeoutMs == null ? (method === "GET" ? 30000 : 60000) : options.timeoutMs;

    const controller = new AbortController();
    const timer = setTimeout(function () {
      controller.abort();
    }, timeoutMs);

    try {
      const fetchOptions = { method: method, signal: controller.signal };
      if (hasBody) {
        fetchOptions.headers = { "Content-Type": "application/json" };
        fetchOptions.body = JSON.stringify(options.body);
      }
      const response = await fetch(url, fetchOptions);
      const payload = await response.json().catch(function () {
        return {};
      });
      if (!response.ok || payload.status === "error") {
        return {
          ok: false,
          status: response.status,
          payload: payload,
          error: parseApiError(response, payload),
        };
      }
      return {
        ok: true,
        status: response.status,
        payload: payload,
        error: null,
      };
    } catch (err) {
      return {
        ok: false,
        status: 0,
        payload: {},
        error: parseApiError(null, null, err),
      };
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * 带超时和 JSON 解析的 POST 请求（对 requestJson 的兼容包装）。
   * @param {string} url
   * @param {Object} body
   * @param {number} [timeoutMs]
   * @returns {Promise<{ok: boolean, status: number, payload: Object, error: Object|null}>}
   */
  async function postJson(url, body, timeoutMs) {
    return requestJson(url, { method: "POST", body: body, timeoutMs: timeoutMs });
  }

  /**
   * 带超时和 JSON 解析的 PUT 请求。
   * @param {string} url
   * @param {Object} body
   * @param {number} [timeoutMs]
   * @returns {Promise<{ok: boolean, status: number, payload: Object, error: Object|null}>}
   */
  async function putJson(url, body, timeoutMs) {
    return requestJson(url, { method: "PUT", body: body, timeoutMs: timeoutMs });
  }

  /**
   * 带超时的 GET 请求（对 requestJson 的兼容包装）。
   * @param {string} url
   * @param {number} [timeoutMs]
   * @returns {Promise<{ok: boolean, status: number, payload: Object, error: Object|null}>}
   */
  async function getJson(url, timeoutMs) {
    return requestJson(url, { method: "GET", timeoutMs: timeoutMs });
  }

  /**
   * 将 ISO 日期格式化为本地短格式。
   * @param {string} iso
   * @returns {string}
   */
  function formatDate(iso) {
    if (!iso) {
      return "--";
    }
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) {
      return String(iso);
    }
    return date.toLocaleDateString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initSidebar("workbench-sidebar", "sidebar-toggle");
  });

  global.KronosWorkbench = {
    textNode: textNode,
    setText: setText,
    clearChildren: clearChildren,
    appendChild: appendChild,
    formatPercent: formatPercent,
    formatNumber: formatNumber,
    formatDate: formatDate,
    parseApiError: parseApiError,
    requestJson: requestJson,
    postJson: postJson,
    putJson: putJson,
    getJson: getJson,
  };
})(window);
