//! 本地 HTTP API：OpenAI 兼容的 `/v1/models` 与 `/v1/chat/completions`。
//! 仅监听 127.0.0.1，供本机其他程序（脚本、插件、其他客户端）调用当前已加载的本地模型。
//! 复用 commands::run_generation，与桌面端共享同一个模型实例与并发保护。

use crate::commands::{self, AppState};
use crate::types::{ChatMessage, InferenceParams};
use axum::extract::State as AxumState;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
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
}

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    /// 兼容 string 与 [{type:"text",text:"..."}] 两种写法。
    #[serde(default)]
    content: Option<Value>,
}

/// 流式输出的消息类型。
enum StreamMsg {
    Delta(String),
    Done,
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

    let history: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: content_to_text(m.content.as_ref()),
        })
        .collect();

    let model_slot = Arc::clone(&state.llama_instance);
    let stop_flag = Arc::clone(&state.generation_stop_flag);
    let is_generating = Arc::clone(&state.is_generating);
    // 请求里指定了 model 就回显该名字，否则回显当前已加载模型名。
    let resp_model = req.model.clone().unwrap_or_else(|| model_name.clone());
    let id = format!("chatcmpl-{}", now_millis());
    let created = now_millis() / 1000;

    if !req.stream {
        let joined = tokio::task::spawn_blocking(move || {
            let mut text = String::new();
            let result = commands::run_generation(
                &model_slot,
                &history,
                &params,
                &stop_flag,
                &mut |delta: &str| {
                    text.push_str(delta);
                    Ok(())
                },
            );
            is_generating.store(false, Ordering::SeqCst);
            (result, text)
        })
        .await;

        return match joined {
            Ok((Ok(_), text)) => Json(json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": resp_model,
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": text },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
            }))
            .into_response(),
            Ok((Err(e), _)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };
    }

    // 流式：阻塞线程逐段发送，SSE 侧转成 OpenAI chunk 格式。
    let (tx, rx) = mpsc::channel::<StreamMsg>(64);
    tokio::task::spawn_blocking(move || {
        let mut sink = |delta: &str| {
            tx.blocking_send(StreamMsg::Delta(delta.to_string()))
                .map_err(|e| e.to_string())
        };
        let result = commands::run_generation(&model_slot, &history, &params, &stop_flag, &mut sink);
        is_generating.store(false, Ordering::SeqCst);
        let last = match result {
            Ok(_) => StreamMsg::Done,
            Err(e) => StreamMsg::Failed(e),
        };
        let _ = tx.blocking_send(last);
    });

    let id_for_stream = id.clone();
    let model_for_stream = resp_model.clone();
    let stream = ReceiverStream::new(rx)
        .map(move |msg| {
            let payload = match msg {
                StreamMsg::Delta(delta) => json!({
                    "id": id_for_stream,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_for_stream,
                    "choices": [{ "index": 0, "delta": { "content": delta }, "finish_reason": null }]
                }),
                StreamMsg::Done => json!({
                    "id": id_for_stream,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_for_stream,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
                }),
                StreamMsg::Failed(e) => json!({
                    "error": { "message": e, "type": "server_error" }
                }),
            };
            Ok::<Event, Infallible>(Event::default().data(payload.to_string()))
        })
        .chain(tokio_stream::once(Ok::<Event, Infallible>(
            Event::default().data("[DONE]"),
        )));

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
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
