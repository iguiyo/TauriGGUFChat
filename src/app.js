// ===== TauriGGUFChat 前端逻辑 =====
// 阶段1：绑定 UI 按钮 -> 调用 tauri command。
// 阶段4：监听 chat_token 事件，逐字渲染流式回复，支持中途停止。
// 阶段5：会话持久化与多会话管理（侧边栏新建/切换/删除）。
// Tauri v2 通过全局 window.__TAURI__ 提供核心 API。

// 获取 Tauri core 的 invoke 方法（适配 v2 全局注入方式）。
function getInvoke() {
  const tauri = window.__TAURI__;
  if (tauri && tauri.core && typeof tauri.core.invoke === "function") {
    return tauri.core.invoke.bind(tauri.core);
  }
  if (tauri && typeof tauri.invoke === "function") {
    return tauri.invoke.bind(tauri);
  }
  return null;
}

const invoke = getInvoke();

// ---- DOM 引用 ----
const modelSelect = document.getElementById("model-select");
const btnImport = document.getElementById("btn-import");
const btnLoad = document.getElementById("btn-load");
const btnUnload = document.getElementById("btn-unload");
const btnDelete = document.getElementById("btn-delete");
const btnSaveConfig = document.getElementById("btn-save-config");
const btnSend = document.getElementById("btn-send");
const btnStop = document.getElementById("btn-stop");
const chatHistory = document.getElementById("chat-history");
const chatInput = document.getElementById("chat-input");
const statusEl = document.getElementById("status");
const overlay = document.getElementById("busy-overlay");
const sessionListEl = document.getElementById("session-list");
const btnNewSession = document.getElementById("btn-new-session");
// 本地 API 控制区
const apiStateEl = document.getElementById("api-state");
const btnApiToggle = document.getElementById("btn-api-toggle");
const apiBaseUrlEl = document.getElementById("api-base-url");

// ---- 运行时状态 ----
// 本地对话消息数组：user / assistant 交替，作为每轮推理的上下文。
let conversation = [];
// 是否正在生成（锁定发送按钮，避免并发推理）。
let isGenerating = false;
// 当前流式输出的 AI 气泡节点。
let currentBubble = null;
// 本轮已流式输出的文本（与后端推送保持同步）。
let streamedText = "";
// 阶段5：全部会话（按 updated_at 倒序）与当前活跃会话。
let sessions = [];
let activeSession = null;

// ===== 工具函数 =====

/** 显示/隐藏全屏加载提示（避免频繁点击）。 */
function setBusy(show, text = "") {
  overlay.classList.toggle("show", show);
  if (text) overlay.querySelector(".box").textContent = text;
}

/** 更新底部状态文本。 */
function setStatus(text) {
  statusEl.textContent = text;
}

/** 切换生成中状态：锁定发送与模型操作，启用停止按钮。 */
function setGenerating(on) {
  isGenerating = on;
  btnSend.disabled = on;
  btnStop.disabled = !on;
  btnLoad.disabled = on;
  btnUnload.disabled = on;
  btnDelete.disabled = on;
}

/** 读取当前表单中的推理参数（与后端 InferenceParams 字段对齐）。 */
function readParams() {
  return {
    n_ctx: Number(document.getElementById("p-ctx").value),
    n_gpu_layers: Number(document.getElementById("p-gpu").value),
    temperature: Number(document.getElementById("p-temp").value),
    max_new_tokens: Number(document.getElementById("p-max").value),
    top_p: Number(document.getElementById("p-topp").value),
  };
}

/** 在聊天容器追加一条消息气泡。 */
function appendMessage(role, content, streaming = false) {
  const div = document.createElement("div");
  div.className = "msg " + role + (streaming ? " streaming" : "");
  div.textContent = content;
  chatHistory.appendChild(div);
  scrollToBottom();
  return div;
}

/** 滚动到底部。 */
function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

// ===== 思考过程（think 段）折叠 =====
// Qwen3 等模型输出形如「<标记>think<标记>…<标记>/think<标记>正文」，
// 这里把思考段拆出来放进可折叠的 <details>，正文照常显示。

/** 是否匹配思考起始标记（<think> / <|think|> 等），命中返回标记结束偏移。 */
function matchThinkOpen(s) {
  const m = /^[^A-Za-z0-9][|｜]?(?:think)(?:ing)?/.exec(s);
  if (!m) return -1;
  // 标记后必须是分隔符，避免把 "<thinking 乱" 这类正文误判。
  const next = s[m[0].length];
  if (next !== undefined && !"><|｜\n\r\t ".includes(next)) return -1;
  let i = m[0].length;
  while (i < s.length && (s[i] === ">" || s[i] === "|" || s[i] === "｜")) i++;
  return i;
}

/** 开头是否可能是被分片截断的起始标记（例如只收到 "<" 或 "<thi"）。 */
function isPartialThinkOpen(s) {
  if (!s) return true;
  if (/^[A-Za-z0-9]/.test(s)) return false;
  const rest = s.slice(1).replace(/^[|｜]/, "");
  if (rest.length >= 8) return false;
  return /^[a-z]*$/.test(rest) && "thinking".startsWith(rest);
}

/** 查找闭合标记（</think> / <|/think|> 等），返回 [起点, 终点)。 */
function findThinkClose(s) {
  for (let j = 1; j + 6 <= s.length; j++) {
    if (!s.startsWith("/think", j)) continue;
    // 标记前的字符必须是分隔符，避免误切正文里的 "/think"。
    if (/[A-Za-z0-9]/.test(s[j - 1])) continue;
    let end = j + 6;
    while (end < s.length && (s[end] === ">" || s[end] === "|" || s[end] === "｜")) end++;
    return [j - 1, end];
  }
  return null;
}

/** 把整段文本拆成思考段与正文。 */
function splitThink(text) {
  const body = text.replace(/^\s+/, "");
  // 标记可能被 token 切开，长度不够时先不下结论，避免闪烁。
  if (isPartialThinkOpen(body)) return { think: "", answer: "", hasThink: false };
  const open = matchThinkOpen(body);
  if (open < 0) return { think: "", answer: text, hasThink: false };
  const rest = body.slice(open);
  const close = findThinkClose(rest);
  if (!close) {
    return { think: rest.replace(/^\s+/, ""), answer: "", hasThink: true, closed: false };
  }
  return {
    think: rest.slice(0, close[0]).trim(),
    answer: rest.slice(close[1]),
    hasThink: true,
    closed: true,
  };
}

/**
 * 渲染 AI 气泡：思考段渲染为可折叠块，正文正常显示。
 * streaming=true 时思考区默认展开，生成结束后自动收起；
 * 用户手动点开/收起过则不再自动改变状态。
 */
function renderAiText(bubble, text, streaming) {
  const p = splitThink(text);
  let block = bubble.querySelector(".think-block");
  let answer = bubble.querySelector(".answer");
  if (!answer) {
    answer = document.createElement("div");
    answer.className = "answer";
    bubble.appendChild(answer);
  }

  if (p.hasThink) {
    if (!block) {
      block = document.createElement("details");
      block.className = "think-block";
      const summary = document.createElement("summary");
      block.appendChild(summary);
      const body = document.createElement("div");
      body.className = "think-body";
      block.appendChild(body);
      bubble.insertBefore(block, answer);
      // 只把用户点击视为「手动切换」（程序改 open 不会触发 click）。
      summary.addEventListener("click", () => {
        bubble.dataset.thinkManual = "1";
      });
    }
    const label = !p.closed && streaming ? "思考中…" : "思考过程";
    block.querySelector("summary").textContent = `${label} （${p.think.length} 字）`;
    block.querySelector(".think-body").textContent = p.think;
    if (!bubble.dataset.thinkManual) block.open = streaming || !p.closed;
  }

  answer.textContent = p.answer;
}

/** 刷新模型下拉列表。 */
async function refreshModelList() {
  if (!invoke) { setStatus("未检测到 Tauri 环境"); return; }
  const models = await invoke("refresh_model_list");
  modelSelect.innerHTML = models.length
    ? models.map(m => `<option value="${m.path}">${m.name}（${(m.size / 1e6).toFixed(0)}MB）</option>`).join("")
    : '<option value="">（暂无模型，请先导入）</option>';
}

// ===== 阶段5：会话管理 =====

/** 时间戳（毫秒）格式化为 MM-DD HH:mm。 */
function formatTime(ms) {
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** 渲染会话列表（标题 + 时间 + 删除按钮，当前会话高亮）。 */
function renderSessionList() {
  sessionListEl.innerHTML = "";
  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = "session-item" + (activeSession && s.id === activeSession.id ? " active" : "");

    const info = document.createElement("div");
    info.className = "session-info";
    const title = document.createElement("div");
    title.className = "session-title";
    title.textContent = s.title || "新会话";
    const time = document.createElement("div");
    time.className = "session-time";
    time.textContent = formatTime(s.updated_at);
    info.appendChild(title);
    info.appendChild(time);

    const del = document.createElement("button");
    del.className = "session-del";
    del.textContent = "×";
    del.title = "删除会话";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      onDeleteSession(s.id);
    });

    item.appendChild(info);
    item.appendChild(del);
    item.addEventListener("click", () => switchSession(s.id));
    sessionListEl.appendChild(item);
  });
}

/** 从后端拉取会话列表并重绘。 */
async function refreshSessions() {
  try {
    sessions = await invoke("list_sessions");
    renderSessionList();
  } catch (e) {
    setStatus("读取会话列表失败：" + e);
  }
}

/** 渲染当前会话的消息列表。 */
function renderConversation() {
  chatHistory.innerHTML = "";
  if (!conversation.length) {
    chatHistory.innerHTML =
      '<div class="chat-placeholder">当前会话暂无消息，开始新对话吧。</div>';
    return;
  }
  conversation.forEach((m) => {
    if (m.role === "user") {
      appendMessage("user", m.content);
    } else {
      // AI 消息：思考段折叠显示（历史消息默认收起）。
      const bubble = document.createElement("div");
      bubble.className = "msg ai";
      chatHistory.appendChild(bubble);
      renderAiText(bubble, m.content, false);
    }
  });
  scrollToBottom();
}

/** 等待生成真正结束（停止标记在采样循环中生效，需要短暂轮询）。 */
function waitGenerationStop(timeout = 5000) {
  return new Promise((resolve) => {
    const started = Date.now();
    const timer = setInterval(() => {
      if (!isGenerating || Date.now() - started > timeout) {
        clearInterval(timer);
        resolve();
      }
    }, 50);
  });
}

/** 把当前 conversation 写回活跃会话并持久化（生成结束 / 首条消息时调用）。 */
async function persistCurrentSession() {
  // 固定目标会话，避免保存过程中用户切换会话导致写错对象。
  const session = activeSession;
  if (!session) return;
  session.messages = conversation.slice();
  session.updated_at = Date.now();
  try {
    await invoke("save_session", { session });
    // 同步列表中的元信息并按 updated_at 重新排序。
    sessions = sessions.filter((s) => s.id !== session.id);
    sessions.push(session);
    sessions.sort((a, b) => b.updated_at - a.updated_at);
    renderSessionList();
  } catch (e) {
    setStatus("保存会话失败：" + e);
  }
}

/** 新建会话并切换过去（若正在生成，先停止推理）。 */
async function createNewSession() {
  if (isGenerating) {
    await invoke("stop_generation");
    await waitGenerationStop();
  }
  const s = await invoke("create_session");
  activeSession = s;
  conversation = [];
  renderConversation();
  await refreshSessions();
  setStatus("已新建会话");
}

/** 切换会话；正在生成时先停止推理，再加载目标会话消息。 */
async function switchSession(id) {
  if (activeSession && activeSession.id === id) return;
  if (isGenerating) {
    await invoke("stop_generation");
    await waitGenerationStop();
  }
  try {
    const s = await invoke("load_session", { sessionId: id });
    activeSession = s;
    conversation = s.messages.slice();
    renderConversation();
    renderSessionList();
    setStatus("已切换会话");
  } catch (e) {
    alert("加载会话失败：" + e);
  }
}

/** 删除会话；若删除的是当前会话则新建一个会话。 */
async function onDeleteSession(id) {
  if (!confirm("确定删除该会话？删除后不可恢复。")) return;
  const deletingActive = activeSession && activeSession.id === id;
  if (deletingActive && isGenerating) {
    await invoke("stop_generation");
    await waitGenerationStop();
  }
  try {
    await invoke("delete_session", { sessionId: id });
    if (deletingActive) {
      await createNewSession();
    } else {
      await refreshSessions();
    }
    setStatus("会话已删除");
  } catch (e) {
    alert("删除会话失败：" + e);
  }
}

// ===== 流式事件 =====

/** 结束本轮生成：同步最终文本、恢复按钮、记录 AI 回复。 */
function finishGeneration(text, isError) {
  // 已收尾过（例如 invoke 异常与 error 事件都到达）时忽略，避免重复弹窗。
  if (!isGenerating && !currentBubble) return;

  const finalText = typeof text === "string" ? text : streamedText;

  if (isError) {
    // 结构化错误：弹窗提示，已输出内容保留在气泡中。
    alert("生成失败：" + finalText);
    setStatus("生成失败");
  } else if (currentBubble) {
    renderAiText(currentBubble, finalText, false);
    // 空回复不入历史（例如加载后立刻停止）。
    if (finalText.trim()) {
      conversation.push({ role: "assistant", content: finalText });
    }
    // 阶段5：生成结束自动持久化当前会话（messages + updated_at）。
    persistCurrentSession();
    setStatus("就绪");
  }

  if (currentBubble) currentBubble.classList.remove("streaming");
  currentBubble = null;
  streamedText = "";
  setGenerating(false);
}

/** 订阅后端 chat_token 事件（页面初始化时调用一次）。 */
function listenChatEvents() {
  const eventApi = window.__TAURI__ && window.__TAURI__.event;
  if (!eventApi || typeof eventApi.listen !== "function") {
    setStatus("未检测到 Tauri 事件系统，流式输出不可用");
    return;
  }
  eventApi.listen("chat_token", (event) => {
    const payload = event.payload || {};
    switch (payload.kind) {
      case "token":
        // 逐字追加到当前 AI 气泡并滚动到底。
        if (!currentBubble) return;
        streamedText += payload.content;
        renderAiText(currentBubble, streamedText, true);
        scrollToBottom();
        break;
      case "end":
        finishGeneration(payload.content, false);
        break;
      case "error":
        finishGeneration(payload.content, true);
        break;
      default:
        break;
    }
  });
}

// ===== 按钮绑定 =====

// 导入 GGUF：弹出说明提示，确认后由后端弹出文件选择对话框，
// 并把所选文件复制一份到软件内部模型库（data_dir/models），源文件可自行删除。
btnImport.addEventListener("click", async () => {
  const agreed = confirm(
    "导入后模型文件将复制到软件内部模型库，原文件可自行删除。\n\n是否选择 GGUF 文件并导入？"
  );
  if (!agreed) return;
  setStatus("请选择要导入的 GGUF 文件…");
  try {
    const item = await invoke("import_model", { path: "C:/path/to/example.gguf" });
    setStatus("导入成功：" + item.name + "（已复制到软件内部模型库，原文件可删除）");
    await refreshModelList();
  } catch (e) {
    setStatus("导入失败：" + e);
  }
});

// 加载选中模型。
btnLoad.addEventListener("click", async () => {
  const path = modelSelect.value;
  if (!path) { setStatus("请先选择一个模型"); return; }
  setBusy(true, "正在加载模型…");
  try {
    await invoke("load_model", { path });
    setStatus("模型加载成功（占位）");
  } catch (e) {
    setStatus("加载失败：" + e);
  } finally {
    setBusy(false);
  }
});

// 卸载模型。
btnUnload.addEventListener("click", async () => {
  try {
    await invoke("unload_model");
    setStatus("模型已卸载（占位）");
  } catch (e) {
    setStatus("卸载失败：" + e);
  }
});

// 删除选中模型。
btnDelete.addEventListener("click", async () => {
  const path = modelSelect.value;
  if (!path) { setStatus("请先选择一个模型"); return; }
  try {
    await invoke("delete_model", { path });
    setStatus("模型已删除（占位）");
    await refreshModelList();
  } catch (e) {
    setStatus("删除失败：" + e);
  }
});

// 保存参数。
btnSaveConfig.addEventListener("click", async () => {
  const config = readParams();
  try {
    await invoke("save_config", { config });
    setStatus("参数已保存");
  } catch (e) {
    setStatus("保存失败：" + e);
  }
});

// 发送消息（阶段4：流式生成）。
async function sendMessage() {
  if (isGenerating) return;
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = "";

  // 阶段5：确保存在活跃会话（理论上初始化时已创建）。
  if (!activeSession) {
    await createNewSession();
  }

  // ① 立刻把用户消息渲染进聊天框，并写入本地历史。
  conversation.push({ role: "user", content: text });
  appendMessage("user", text);

  // 阶段5：首条用户消息自动生成会话标题（前 20 字符）并保存。
  if (conversation.length === 1) {
    const title = text.slice(0, 20);
    activeSession.title = title;
    await persistCurrentSession();
  }

  // ② 预先创建空 AI 气泡，后续 token 持续追加。
  currentBubble = appendMessage("ai", "", true);
  streamedText = "";
  setGenerating(true);
  setStatus("正在生成…");

  // ③ 调起后端流式推理；结果由 chat_token 事件驱动。
  try {
    await invoke("send_chat", {
      history: conversation,
      params: readParams(),
    });
  } catch (e) {
    // 后端在生成前的错误（未加载模型、参数非法等）：走统一的错误收尾。
    if (isGenerating) finishGeneration(String(e), true);
  }
}

// 停止生成：只置位后端终止标志，已输出内容保留。
btnStop.addEventListener("click", async () => {
  if (!isGenerating) return;
  setStatus("正在停止…");
  try {
    await invoke("stop_generation");
  } catch (e) {
    setStatus("停止失败：" + e);
  }
});

btnSend.addEventListener("click", sendMessage);

// 阶段5：新建会话。
btnNewSession.addEventListener("click", async () => {
  try {
    await createNewSession();
  } catch (e) {
    setStatus("新建会话失败：" + e);
  }
});

// Enter 发送，Shift+Enter 换行。
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ===== 本地 API 开关 =====

/** 根据后端返回的状态刷新 API 区域显示。 */
function renderApiStatus(s) {
  if (!s) return;
  apiStateEl.textContent = s.running ? "运行中" : "已关闭";
  apiStateEl.classList.toggle("on", !!s.running);
  apiStateEl.classList.toggle("off", !s.running);
  btnApiToggle.textContent = s.running ? "关闭 API" : "开启 API";
  apiBaseUrlEl.textContent = s.base_url;
}

/** 查询一次 API 状态（初始化与开关后调用）。 */
async function refreshApiStatus() {
  try {
    renderApiStatus(await invoke("api_status"));
  } catch (e) {
    setStatus("获取 API 状态失败：" + e);
  }
}

// 开启 / 关闭本地 API。
btnApiToggle.addEventListener("click", async () => {
  btnApiToggle.disabled = true;
  try {
    const current = await invoke("api_status");
    const s = current.running ? await invoke("stop_api") : await invoke("start_api");
    renderApiStatus(s);
    setStatus(s.running ? "本地 API 已开启" : "本地 API 已关闭");
  } catch (e) {
    setStatus("API 操作失败：" + e);
    await refreshApiStatus();
  } finally {
    btnApiToggle.disabled = false;
  }
});

// ===== 初始化 =====
(async function init() {
  try {
    const config = await invoke("get_config");
    if (config) {
      document.getElementById("p-ctx").value = config.n_ctx;
      document.getElementById("p-gpu").value = config.n_gpu_layers;
      document.getElementById("p-temp").value = config.temperature;
      document.getElementById("p-max").value = config.max_new_tokens;
      document.getElementById("p-topp").value = config.top_p;
    }
    await refreshModelList();
    // 阶段5：加载会话列表；无会话时自动创建一个，否则打开最近更新的会话。
    await refreshSessions();
    if (sessions.length === 0) {
      await createNewSession();
    } else {
      const latest = sessions[0];
      activeSession = latest;
      conversation = latest.messages.slice();
      renderConversation();
      renderSessionList();
    }
    // 页面初始化即订阅流式事件（必须早于首次发送）。
    listenChatEvents();
    // 同步本地 API 的当前状态（后端默认在启动时开启）。
    await refreshApiStatus();
    setStatus("就绪");
  } catch (e) {
    setStatus("初始化失败：" + e);
  }
})();