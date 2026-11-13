//! 本地 HTTP API：OpenAI 兼容的 `/v1/models` 与 `/v1/chat/completions`。
//! 仅监听 127.0.0.1，供本机其他程序（脚本、插件、其他客户端）调用当前已加载的本地模型。
//! 复用 commands::run_generation，与桌面端共享同一个模型实例与并发保护。

use crate::commands::{self, AppState};
use crate::types::InferenceParams;
use axum::extract::State as AxumState;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use llama_cpp_2::model::LlamaModel;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;

/// 本地 API 端口（只绑定 127.0.0.1，不对外网暴露）。
pub const API_PORT: u16 = 18080;

#[derive(Clone)]
struct ApiCtx {
    app: AppHandle,
}

/// 本地 API 的运行状态，由 Tauri 管理，供前端「开启/关闭」。
pub struct ApiControl {
    /// 关闭信号发送端；Some 表示服务线程存活。
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    running: Arc<AtomicBool>,
}

/// 返回给前端的 API 状态。
#[derive(serde::Serialize)]
pub struct ApiStatus {
    pub running: bool,
    pub port: u16,
    pub base_url: String,
}

impl ApiControl {
    pub fn new() -> Self {
        Self {
            shutdown: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    fn status(&self) -> ApiStatus {
        ApiStatus {
            running: self.running.load(Ordering::SeqCst),
            port: API_PORT,
            base_url: format!("http://127.0.0.1:{}/v1", API_PORT),
        }
    }

    /// 启动服务。端口占用会立即返回错误，避免界面显示「已开启」但实际没监听。
    pub fn start(&self, app: AppHandle) -> Result<ApiStatus, String> {
        if self.running.load(Ordering::SeqCst) {
            return Err("本地 API 已在运行".to_string());
        }
        let std_listener = bind_with_retry()?;
        let (tx, rx) = oneshot::channel();
        *self.shutdown.lock().unwrap() = Some(tx);
        self.running.store(true, Ordering::SeqCst);

        let running = Arc::clone(&self.running);
        std::thread::Builder::new()
            .name("local-api".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        log::error!("本地 API 运行时创建失败: {}", e);
                        running.store(false, Ordering::SeqCst);
                        return;
                    }
                };
                rt.block_on(async move {
                    let listener = match tokio::net::TcpListener::from_std(std_listener) {
                        Ok(l) => l,
                        Err(e) => {
                            log::error!("本地 API 监听器创建失败: {}", e);
                            running.store(false, Ordering::SeqCst);
                            return;
                        }
                    };
                    log::info!("本地 API 已启动: http://127.0.0.1:{}/v1", API_PORT);
                    if let Err(e) = axum::serve(listener, build_router(app))
                        .with_graceful_shutdown(async move {
                            let _ = rx.await;
                        })
                        .await
                    {
                        log::error!("本地 API 服务退出: {}", e);
                    }
                    running.store(false, Ordering::SeqCst);
                    log::info!("本地 API 已停止");
                });
            })
            .map_err(|e| format!("启动 API 线程失败: {}", e))?;

        Ok(self.status())
    }

    /// 停止服务：发出关闭信号，服务线程在完成当前请求后退出。
    pub fn stop(&self) -> Result<ApiStatus, String> {
        if !self.running.load(Ordering::SeqCst) {
            return Err("本地 API 未在运行".to_string());
        }
        if let Some(tx) = self.shutdown.lock().unwrap().take() {
            let _ = tx.send(());
        }
        self.running.store(false, Ordering::SeqCst);
        Ok(self.status())
    }
}

/// 绑定 127.0.0.1:API_PORT，带短重试（刚关闭后端口可能尚未释放）。
fn bind_with_retry() -> Result<std::net::TcpListener, String> {
    let mut last = String::new();
    for _ in 0..10 {
        match std::net::TcpListener::bind(("127.0.0.1", API_PORT)) {
            Ok(listener) => {
                listener
                    .set_nonblocking(true)
                    .map_err(|e| format!("设置非阻塞失败: {}", e))?;
                return Ok(listener);
            }
            Err(e) => {
                last = e.to_string();
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    Err(format!("端口 {} 无法监听: {}", API_PORT, last))
}

fn build_router(app: AppHandle) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(ApiCtx { app })
}

/// 查询本地 API 运行状态。
#[tauri::command]
pub fn api_status(ctl: tauri::State<ApiControl>) -> ApiStatus {
    ctl.status()
}

/// 开启本地 API。
#[tauri::command]
pub fn start_api(app: AppHandle, ctl: tauri::State<ApiControl>) -> Result<ApiStatus, String> {
    ctl.start(app)
}

/// 关闭本地 API。
#[tauri::command]
pub fn stop_api(ctl: tauri::State<ApiControl>) -> Result<ApiStatus, String> {
    ctl.stop()
}

async fn health() -> Response {
    Json(json!({ "status": "ok" })).into_response()
}

/// GET /v1/models：返回模型库中的全部模型（与桌面端模型列表同源）。
async fn list_models(AxumState(ctx): AxumState<ApiCtx>) -> Response {
    let state = ctx.app.state::<AppState>();
    let models = match crate::fs_service::read_model_list(&state.base_dir) {
        Ok(list) => list,
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
    };
    let data: Vec<Value> = models
        .iter()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "owned_by": "local",
                "created": 0,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data })).into_response()
}

#[derive(Deserialize)]
struct ChatRequest {
    /// 模型名（可选，当前实现始终使用已加载的模型；传了也仅做透传展示）。
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    /// OpenAI 函数调用定义。
    #[serde(default)]
    tools: Option<Vec<Value>>,
    /// "auto" / "none" / {"type":"function","function":{"name":"..."}}
    #[serde(default)]
    tool_choice: Option<Value>,
    /// 流式参数（目前只关心 include_usage）。
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    /// 兼容 string 与 [{type:"text",text:"..."}] 两种写法；tool_calls 消息可为 null。
    #[serde(default)]
    content: Option<Value>,
    /// assistant 历史中上一次的工具调用。
    #[serde(default)]
    tool_calls: Option<Vec<ApiToolCall>>,
    /// role=tool 时对应哪次调用（仅回显，不进 prompt）。
    #[serde(default)]
    #[allow(dead_code)]
    tool_call_id: Option<String>,
}

/// 工具调用（请求历史与响应共用）。
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct ApiToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(default = "default_call_type", rename = "type")]
    r#type: String,
    function: ApiFunctionCall,
}

fn default_call_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct ApiFunctionCall {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

/// 流式输出的消息类型。
enum StreamMsg {
    Delta(String),
    /// 工具调用：一次性下发完整的 tool_calls 增量。
    ToolCalls(Vec<ApiToolCall>),
    /// 生成结束，携带 finish_reason（stop / tool_calls）。
    Done(String),
    Failed(String),
}

/// POST /v1/chat/completions：OpenAI 兼容，支持 stream=true（SSE 增量）与 stream=false（整段返回）。
async fn chat_completions(
    AxumState(ctx): AxumState<ApiCtx>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let state = ctx.app.state::<AppState>();

    let model_name = match state.current_loaded_model.lock().unwrap().as_ref() {
        Some(m) => m.name.clone(),
        None => {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "尚未加载模型，请先在桌面端点击「加载选中模型」",
            )
        }
    };

    // 与桌面端共享并发保护：同一时间只跑一个推理任务。
    if state
        .is_generating
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return api_error(StatusCode::TOO_MANY_REQUESTS, "正在生成中，请稍后重试");
    }
    state.generation_stop_flag.store(false, Ordering::SeqCst);

    let cfg = state.current_config.lock().unwrap().clone();
    let params = InferenceParams {
        n_ctx: cfg.n_ctx,
        n_gpu_layers: cfg.n_gpu_layers,
        temperature: req.temperature.unwrap_or(cfg.temperature),
        max_new_tokens: req.max_tokens.unwrap_or(cfg.max_new_tokens),
        top_p: req.top_p.unwrap_or(cfg.top_p),
    };

    // ---- 组装 prompt：ChatML +（可选）tools 注入 ----
    // tool_choice="none" 时不注入工具定义，按普通对话处理。
    let tools_enabled = req
        .tools
        .as_ref()
        .map_or(false, |t| !t.is_empty())
        && !matches!(&req.tool_choice, Some(Value::String(s)) if s == "none");
    let prompt = build_api_prompt(
        &req.messages,
        if tools_enabled { req.tools.as_deref() } else { None },
        req.tool_choice.as_ref(),
        is_qwen3(&model_name),
    );

    let model_slot = Arc::clone(&state.llama_instance);
    let stop_flag = Arc::clone(&state.generation_stop_flag);
    let is_generating = Arc::clone(&state.is_generating);
    // 请求里指定了 model 就回显该名字，否则回显当前已加载模型名。
    let resp_model = req.model.clone().unwrap_or_else(|| model_name.clone());
    let id = format!("chatcmpl-{}", now_millis());
    let created = now_millis() / 1000;

    if !req.stream {
        let joined = tokio::task::spawn_blocking(move || {
            let (result, content, tool_text) = generate_with_filters(
                &model_slot,
                &prompt,
                &params,
                &stop_flag,
                &mut |_delta: &str| Ok(()),
            );
            is_generating.store(false, Ordering::SeqCst);
            (result, content, tool_text)
        })
        .await;

        return match joined {
            Ok((Ok(_), content, tool_text)) => {
                let calls = parse_tool_calls(&tool_text);
                let finish_reason = if calls.is_empty() { "stop" } else { "tool_calls" };
                // 没有解析出工具调用时，把原文原样作为 content 返回，避免丢内容。
                let content = if calls.is_empty() {
                    format!("{}{}", content, tool_text)
                } else {
                    content
                };
                let mut message = json!({ "role": "assistant", "content": content });
                if !calls.is_empty() {
                    message["tool_calls"] = serde_json::to_value(&calls).unwrap_or(Value::Null);
                }
                Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": resp_model,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    }],
                    "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
                }))
                .into_response()
            }
            Ok((Err(e), _, _)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };
    }

    // 流式：阻塞线程逐段发送，SSE 侧转成 OpenAI chunk 格式。
    let include_usage = req.stream_options.as_ref().map_or(false, |o| o.include_usage);
    let (tx, rx) = mpsc::channel::<StreamMsg>(64);
    tokio::task::spawn_blocking(move || {
        let (result, _content, tool_text) = generate_with_filters(
            &model_slot,
            &prompt,
            &params,
            &stop_flag,
            &mut |delta: &str| {
                tx.blocking_send(StreamMsg::Delta(delta.to_string()))
                    .map_err(|e| e.to_string())
            },
        );
        is_generating.store(false, Ordering::SeqCst);
        let last = match result {
            Ok(_) => {
                let calls = parse_tool_calls(&tool_text);
                if calls.is_empty() {
                    StreamMsg::Done("stop".to_string())
                } else {
                    // 工具调用一次性完整下发，随后用 tool_calls 作为结束原因。
                    let _ = tx.blocking_send(StreamMsg::ToolCalls(calls));
                    StreamMsg::Done("tool_calls".to_string())
                }
            }
            Err(e) => StreamMsg::Failed(e),
        };
        let _ = tx.blocking_send(last);
    });

    let id_for_stream = id.clone();
    let model_for_stream = resp_model.clone();
    let usage_id = id.clone();
    let usage_model = resp_model.clone();
    // 首个 chunk 先声明 role：部分客户端只读 delta.role 建消息，缺了会显示空消息。
    let role_chunk = tokio_stream::once(Ok::<Event, Infallible>(
        Event::default().data(
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": resp_model,
                "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "" }, "finish_reason": null }]
            })
            .to_string(),
        ),
    ));

    let stream = role_chunk
        .chain(
            ReceiverStream::new(rx)
                .map(move |msg| {
                    let payload = match msg {
                        StreamMsg::Delta(delta) => json!({
                            "id": id_for_stream,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_for_stream,
                            "choices": [{ "index": 0, "delta": { "content": delta }, "finish_reason": null }]
                        }),
                        StreamMsg::ToolCalls(calls) => json!({
                            "id": id_for_stream,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_for_stream,
                            "choices": [{ "index": 0, "delta": { "tool_calls": tool_calls_delta(&calls) }, "finish_reason": null }]
                        }),
                        StreamMsg::Done(reason) => json!({
                            "id": id_for_stream,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_for_stream,
                            "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }]
                        }),
                        StreamMsg::Failed(e) => json!({
                            "error": { "message": e, "type": "server_error" }
                        }),
                    };
                    Ok::<Event, Infallible>(Event::default().data(payload.to_string()))
                })
                .chain(tokio_stream::iter(include_usage.then(|| {
                    Ok::<Event, Infallible>(Event::default().data(
                        json!({
                            "id": usage_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": usage_model,
                            "choices": [],
                            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
                        })
                        .to_string(),
                    ))
                })))
                .chain(tokio_stream::once(Ok::<Event, Infallible>(
                    Event::default().data("[DONE]"),
                ))),
        );

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// 组装 ChatML prompt（Qwen3 / Qwen2.5 原生工具调用格式）。
/// - 有 tools 时，函数签名放进 system 的 `<tools>` 块，并要求用 `<tool_call>` 返回
/// - assistant 历史里的 tool_calls 渲染成 `<tool_call>{...}</tool_call>`
/// - role=tool 的结果渲染成 `<tool_response>...</tool_response>`
fn build_api_prompt(
    messages: &[ApiMessage],
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
    disable_thinking: bool,
) -> String {
    let mut prompt = String::new();

    // 1) 合并所有 system 消息（OpenAI 客户端通常只在首条给出）。
    let system_text = messages
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| content_to_text(m.content.as_ref()))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    if !system_text.is_empty() || tools.is_some() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(&system_text);
        if let Some(tools) = tools {
            if !system_text.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(&tools_block(tools, tool_choice));
        }
        if disable_thinking {
            prompt.push_str("\n\nAnswer directly without any thinking or reasoning process.");
        }
        prompt.push_str("<|im_end|>\n");
    }

    // 2) 其余消息按原顺序渲染。
    for msg in messages.iter().filter(|m| m.role != "system") {
        let content = content_to_text(msg.content.as_ref());
        match msg.role.as_str() {
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&content);
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        prompt.push_str("\n<tool_call>\n");
                        prompt.push_str(&format!(
                            "{{\"name\": \"{}\", \"arguments\": {}}}",
                            call.function.name, call.function.arguments
                        ));
                        prompt.push_str("\n</tool_call>");
                    }
                }
                prompt.push_str("<|im_end|>\n");
            }
            "tool" => {
                prompt.push_str("<|im_start|>user\n<tool_response>\n");
                prompt.push_str(&content);
                prompt.push_str("\n</tool_response><|im_end|>\n");
            }
            _ => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&content);
                prompt.push_str("<|im_end|>\n");
            }
        }
    }

    // 预留 assistant 起始标记，模型从这里续写。
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// tools 定义块（文案与 Qwen3 官方 chat template 一致）。
fn tools_block(tools: &[Value], tool_choice: Option<&Value>) -> String {
    let mut block = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        block.push('\n');
        block.push_str(&tool.to_string());
    }
    block.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    // tool_choice 指定了具体函数时给出提示。
    if let Some(name) = tool_choice
        .and_then(|c| c.get("function"))
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
    {
        block.push_str(&format!("\n\nAlways use the function \"{}\".", name));
    }
    block
}

/// 粗略判断是否 Qwen3 系列（用于关闭思考段）。
fn is_qwen3(model_name: &str) -> bool {
    model_name.to_ascii_lowercase().contains("qwen3")
}

/// 在阻塞线程执行生成，并对输出做「思考段过滤 + 工具调用切分」。
/// 返回：(生成结果, 已下发的正文, `<tool_call>` 起的工具调用原文)。
fn generate_with_filters(
    model_slot: &Mutex<Option<LlamaModel>>,
    prompt: &str,
    params: &InferenceParams,
    stop_flag: &AtomicBool,
    on_text: &mut dyn FnMut(&str) -> Result<(), String>,
) -> (Result<String, String>, String, String) {
    let mut think = ThinkFilter::new();
    let mut gate = ToolGate::new();
    let mut content = String::new();

    let result = commands::run_generation_with_prompt(
        model_slot,
        prompt,
        params,
        stop_flag,
        &mut |delta: &str| {
            let text = think.feed(delta);
            let out = gate.feed(&text);
            if !out.is_empty() {
                content.push_str(&out);
                on_text(&out)?;
            }
            Ok(())
        },
    );

    // 收尾：刷出过滤器里剩余的正文。
    let rest = think.finish();
    let out = gate.feed(&rest);
    if !out.is_empty() {
        content.push_str(&out);
        let _ = on_text(&out);
    }
    let (tail, tool_text) = gate.finish();
    if !tail.is_empty() {
        content.push_str(&tail);
        let _ = on_text(&tail);
    }

    (result, content, tool_text)
}

/// 过滤 Qwen3 等模型的思考段（` thinking ... <｜end▁of▁thinking｜>`）。
/// 只在生成开头判断：不确定时先缓冲，确认不是思考段后原样输出。
struct ThinkFilter {
    buf: String,
    /// 0=尚未确定 1=正在丢弃思考段 2=正常输出
    state: u8,
}

impl ThinkFilter {
    fn new() -> Self {
        Self {
            buf: String::new(),
            state: 0,
        }
    }

    fn feed(&mut self, delta: &str) -> String {
        self.buf.push_str(delta);
        let mut out = String::new();
        loop {
            match self.state {
                0 => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        // 纯空白：等更多内容再判断（最多等 8 字节，避免无限缓冲）。
                        if self.buf.len() >= 8 {
                            self.state = 2;
                            continue;
                        }
                        break;
                    }
                    if let Some(prefix_len) = think_open_len(trimmed) {
                        let lead = self.buf.len() - trimmed.len();
                        self.buf.drain(..lead + prefix_len);
                        self.state = 1;
                        continue;
                    }
                    if is_partial_think_open(trimmed) {
                        break;
                    }
                    self.state = 2;
                    continue;
                }
                1 => {
                    if let Some((start, end)) = find_think_close(&self.buf) {
                        self.buf.drain(..start + end);
                        self.state = 2;
                        continue;
                    }
                    // 只保留可能是半个闭合标记的尾巴。
                    if self.buf.len() > 16 {
                        let cut = floor_char_boundary(&self.buf, self.buf.len() - 16);
                        self.buf.drain(..cut);
                    }
                    break;
                }
                _ => {
                    out.push_str(&self.buf);
                    self.buf.clear();
                    break;
                }
            }
        }
        out
    }

    /// 生成结束时的剩余文本。
    fn finish(&mut self) -> String {
        let mut out = String::new();
        match self.state {
            2 => out.push_str(&self.buf),
            // 尚未确定：按正文输出（丢掉前导空白）
            0 => out.push_str(self.buf.trim_start()),
            // 思考段未闭合：整体丢弃
            _ => {}
        }
        self.buf.clear();
        out
    }
}

/// 判断文本开头是否为思考段起始标记（任意单个分界字符 + "think"）；
/// 命中时返回 `think` 前缀的字节长度。
fn think_open_len(s: &str) -> Option<usize> {
    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    let rest = &s[first.len_utf8()..];
    if rest.starts_with("think") && first != 't' {
        Some(first.len_utf8())
    } else {
        None
    }
}

/// 开头可能是被分片截断的 "<xthink"（例如只收到 "<" 或 "<th"）。
fn is_partial_think_open(s: &str) -> bool {
    let mut chars = s.char_indices();
    let Some((_, first)) = chars.next() else {
        return true;
    };
    if first == 't' {
        return false;
    }
    let rest = &s[first.len_utf8()..];
    if rest.is_empty() {
        return s.len() <= 4;
    }
    "think".starts_with(rest) && rest.chars().all(|c| c.is_ascii_alphabetic())
}

/// 查找 `</think>` 形式的闭合标记（任意单个分界字符 + "/think"）。
/// 返回 (标记起点相对 buf 的偏移, 标记总字节长度)。
fn find_think_close(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    for (i, _) in s.char_indices() {
        let rest = &s[i..];
        let mut chars = rest.char_indices();
        let Some((_, first)) = chars.next() else {
            continue;
        };
        // 距结尾不足一个标记长度时放弃，交由下一次 feed 继续判断。
        if bytes.len() - i < 7 {
            return None;
        }
        let after = &rest[first.len_utf8()..];
        if after.starts_with("/think") {
            return Some((i, first.len_utf8() + "/think".len()));
        }
    }
    None
}

/// 字节下标向下取整到字符边界。
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// 把正文与工具调用分开：正常文本照常下发，出现 `<tool_call>` 后停止输出正文。
struct ToolGate {
    raw: String,
    emitted: usize,
    tool_start: Option<usize>,
}

impl ToolGate {
    fn new() -> Self {
        Self {
            raw: String::new(),
            emitted: 0,
            tool_start: None,
        }
    }

    /// 返回本次可安全输出的文本增量。
    fn feed(&mut self, delta: &str) -> String {
        if delta.is_empty() {
            return String::new();
        }
        self.raw.push_str(delta);
        if self.tool_start.is_none() {
            if let Some(i) = self.raw.find(TOOL_OPEN) {
                self.tool_start = Some(i);
            }
        }
        let limit = match self.tool_start {
            Some(i) => i,
            // 保留标记长度的尾巴，避免把跨分片的 <tool_call> 提前输出。
            None => self.raw.len().saturating_sub(TOOL_OPEN.len() - 1),
        };
        let end = floor_char_boundary(&self.raw, limit);
        if end > self.emitted {
            let out = self.raw[self.emitted..end].to_string();
            self.emitted = end;
            out
        } else {
            String::new()
        }
    }

    /// 收尾：返回（剩余正文, `<tool_call>` 起的原文）。
    fn finish(&mut self) -> (String, String) {
        let start = self.tool_start.unwrap_or(self.raw.len()).min(self.raw.len());
        let end = floor_char_boundary(&self.raw, start);
        let tail = if end > self.emitted {
            self.raw[self.emitted..end].to_string()
        } else {
            String::new()
        };
        let tool_text = self.raw[start..].to_string();
        (tail, tool_text)
    }
}

/// 解析模型输出的工具调用。
fn parse_tool_calls(text: &str) -> Vec<ApiToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;

    while let Some(start) = rest.find(TOOL_OPEN) {
        let after = &rest[start + TOOL_OPEN.len()..];
        let (body, next) = match after.find(TOOL_CLOSE) {
            Some(end) => (&after[..end], &after[end + TOOL_CLOSE.len()..]),
            None => (after, ""),
        };
        if let Some(call) = build_tool_call(body) {
            calls.push(call);
        }
        rest = next;
    }

    // 没有 <tool_call> 标签时，兼容模型直接输出单个 JSON 的情况。
    if calls.is_empty() {
        if let Some(call) = build_tool_call(text) {
            calls.push(call);
        }
    }
    calls
}

/// 从一段文本里抽出 {"name":..,"arguments":..} 并转成 OpenAI tool_call。
fn build_tool_call(body: &str) -> Option<ApiToolCall> {
    let start = body.find('{')?;
    let end = body.rfind('}')?;
    if end <= start {
        return None;
    }
    let value: Value = serde_json::from_str(&body[start..=end]).ok()?;
    let function = value.get("function").unwrap_or(&value);
    let name = function.get("name").and_then(|n| n.as_str())?.to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = function
        .get("arguments")
        .or_else(|| function.get("parameters"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = match arguments {
        Value::String(s) => s,
        other => other.to_string(),
    };
    Some(ApiToolCall {
        id: Some(next_tool_call_id()),
        r#type: default_call_type(),
        function: ApiFunctionCall { name, arguments },
    })
}

/// 递增的工具调用 id。
fn next_tool_call_id() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!(
        "call_{}_{}",
        now_millis(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    )
}

/// 流式增量格式：必须带 index。
fn tool_calls_delta(calls: &[ApiToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    "index": index,
                    "id": call.id.clone().unwrap_or_default(),
                    "type": call.r#type,
                    "function": { "name": call.function.name, "arguments": call.function.arguments }
                })
            })
            .collect(),
    )
}


/// 把 OpenAI 的 content 字段统一成纯文本。
fn content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn api_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": "invalid_request_error" } })),
    )
        .into_response()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}
