//! Tauri command 实现。
//! 阶段2：实现真实文件 IO（导入 GGUF、刷新列表、删除模型）。
//! 阶段3：接入 llama-cpp-2，实现真实模型加载 / 卸载。
//! 阶段4：实现流式对话推理（ChatML 上下文组装 + token 逐字推送 + 可中断）。

use crate::fs_service;
use crate::types::{AppConfig, ChatMessage, ChatTokenEvent, InferenceParams, ModelItem, Session};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, State};

/// llama 后端全局单例。
/// llama.cpp 的后端只能被初始化一次（否则报 BackendAlreadyInitialized），
/// 故使用 OnceLock 缓存，后续所有模型加载 / 推理共用同一个后端。
static LLAMA_BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

/// 获取已初始化的 llama 后端。
/// OnceLock 保存 `Result`：后端初始化可能失败（驱动/显存问题），
/// 失败时返回结构化错误文案，而不会 panic。
fn backend() -> Result<&'static LlamaBackend, String> {
    let init = LLAMA_BACKEND.get_or_init(|| {
        LlamaBackend::init().map_err(|e| format!("初始化 llama 后端失败: {}", e))
    });
    init.as_ref().map_err(|e| e.clone())
}

/// 应用共享状态。
pub struct AppState {
    /// 基础目录：models/、config.json、chat_history/ 均位于此。
    /// 由 tauri path resolver 在启动时解析并注入。
    pub base_dir: PathBuf,
    /// 当前加载的模型标记（供前端展示与清单状态同步）。
    pub current_loaded_model: Mutex<Option<ModelItem>>,
    /// 当前推理参数。
    pub current_config: Mutex<AppConfig>,
    /// 全局唯一 llama 实例。
    /// 约束：同一时间仅允许持有一个模型实例；加载新模型前必须先释放旧实例。
    /// 用 Arc 包裹，便于把模型槽 move 进阻塞推理线程（spawn_blocking 要求 'static）。
    pub llama_instance: Arc<Mutex<Option<LlamaModel>>>,
    /// 阶段4：流式生成终止标志（由 stop_generation 置位，采样循环检测后中断）。
    pub generation_stop_flag: Arc<AtomicBool>,
    /// 阶段4：是否正在生成（并发保护；同一时间只允许一个推理任务）。
    pub is_generating: Arc<AtomicBool>,
}

impl AppState {
    /// 创建状态；base_dir 由 tauri path resolver 提供。
    pub fn new(base_dir: PathBuf) -> Self {
        // 初始化配置文件（首次运行自动创建模板）。
        let _ = fs_service::load_config(&base_dir);
        Self {
            base_dir: base_dir.clone(),
            current_loaded_model: Mutex::new(None),
            current_config: Mutex::new(crate::fs_service::load_config(&base_dir)),
            llama_instance: Arc::new(Mutex::new(None)),
            generation_stop_flag: Arc::new(AtomicBool::new(false)),
            is_generating: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// 导入 GGUF：选择文件 -> 复制到 models/ -> 追加清单。
/// 仅允许 *.gguf 文件。
/// 说明：`path` 参数仅为兼容阶段1前端调用而保留，实际忽略，
/// 文件选择由本命令内部弹出对话框完成。
#[tauri::command]
pub async fn import_model(
    state: State<'_, AppState>,
    path: Option<String>,
) -> Result<ModelItem, String> {
    let base_dir = state.base_dir.clone();
    let _ = path;

    // 1. 打开文件选择对话框，仅允许 gguf。
    //    使用 rfd 同步对话框（rfd 通过 dialog 插件进入依赖树）。
    let src = tokio::task::spawn_blocking(|| {
        let selected = rfd::FileDialog::new()
            .set_title("选择要导入的 GGUF 模型文件")
            .add_filter("GGUF 模型", &["gguf"])
            .pick_file();
        selected
    })
    .await
    .map_err(|e| format!("打开文件对话框失败: {}", e))?;

    // 2. 用户取消则返回空。
    let src = match src {
        Some(p) => p,
        None => return Err("用户取消了文件选择".to_string()),
    };

    // 3. 复制前检查项目内是否已有同名文件。
    let file_name = fs_service::file_name_from(&src);
    if !file_name.to_lowercase().ends_with(".gguf") {
        return Err("只能导入 .gguf 格式的模型文件".to_string());
    }
    fs_service::ensure_unique_filename(&base_dir, &file_name)?;

    // 4. 复制源文件到 models/。
    let dst_dir = base_dir.join("models");
    std::fs::create_dir_all(&dst_dir).map_err(|e| format!("创建 models 目录失败: {}", e))?;
    let dst = dst_dir.join(&file_name);
    fs_service::copy_file(&src, &dst)?;

    // 5. 计算大小并构造记录。
    let size = fs_service::file_size(&dst);
    let item = ModelItem {
        name: file_name.trim_end_matches(".gguf").to_string(),
        file_name: file_name.clone(),
        path: dst.to_string_lossy().to_string(),
        size,
        loaded: false,
    };

    // 6. 追加到清单并保存。
    let mut models = fs_service::read_model_list(&base_dir)?;
    // 兜底查重（避免并发写入导致重名）。
    if models.iter().any(|m| m.file_name == file_name) {
        let _ = fs_service::delete_file(&dst);
        return Err(format!("模型已存在: {}", file_name));
    }
    models.push(item.clone());
    fs_service::write_model_list(&base_dir, &models)?;

    Ok(item)
}

/// 刷新模型清单，返回全部已导入模型。
#[tauri::command]
pub fn refresh_model_list(state: State<'_, AppState>) -> Result<Vec<ModelItem>, String> {
    let models = fs_service::read_model_list(&state.base_dir)?;
    Ok(models)
}

/// 加载选中的模型（阶段3：真实 llama 加载）。
/// 入参：
///   - `path`：模型在清单中的 path 或 file_name（前端原有调用，保持不变）。
///   - `model`：可选的完整模型信息（对应阶段3规格的 ModelItem，缺省则按 path 从清单查）。
///   - `config`：可选的推理参数（对应阶段3规格的 AppConfig，缺省则使用已保存配置）。
/// 流程：
///   1. 若已有 llama 实例，先释放旧实例（置 None，llama.cpp 资源自动回收）。
///   2. 拼接 models/ 下的完整路径。
///   3. 在阻塞线程（tokio::spawn_blocking）中调用 LlamaModel::load_from_file。
///      - 加载是阻塞重操作，放入阻塞线程可避免卡死 Tauri 主线程/UI。
///   4. 加载成功写出全局单例并同步标记；失败返回结构化错误供前端弹窗。
#[tauri::command]
pub async fn load_model(
    state: State<'_, AppState>,
    path: String,
    model: Option<ModelItem>,
    config: Option<AppConfig>,
) -> Result<ModelItem, String> {
    let base_dir = state.base_dir.clone();

    // 1. 解析目标模型：优先使用前端传入的 model，否则按 path/file_name 从清单查找。
    let item = match model {
        Some(m) => m,
        None => {
            let list = fs_service::read_model_list(&base_dir)?;
            list.iter()
                .find(|m| m.path == path || m.file_name == path)
                .cloned()
                .ok_or_else(|| "模型未在清单中找到".to_string())?
        }
    };

    // 2. 推理参数：优先使用前端传入 config，否则回退到已保存配置。
    //    转换出 InferenceParams 以与前端表单字段对齐（n_ctx 等供阶段4构造上下文使用）。
    let cfg = match config {
        Some(c) => c,
        None => state.current_config.lock().unwrap().clone(),
    };
    let infer: InferenceParams = cfg.into();
    let n_gpu_layers = infer.n_gpu_layers;

    // 3. 判断目标逻辑有效，拼接 models/ 下的完整路径。
    let model_path = base_dir.join("models").join(&item.file_name);
    if !model_path.exists() {
        return Err(format!("模型文件不存在: {}", model_path.display()));
    }

    // 4. 取出旧实例并随阻塞任务一起释放（重量级析构移到阻塞线程，避免卡 UI）。
    let old = { state.llama_instance.lock().unwrap().take() };

    // 5. 阻塞线程内完成真实 llama 加载。
    let loaded: Result<LlamaModel, String> = tokio::task::spawn_blocking(move || {
        // 先释放旧实例（如存在），保证全局严格单例。
        drop(old);
        // 获取 llama 后端（全局单例）。
        let backend = backend()?;
        // 构造模型参数：GPU 卸载层数读取配置。
        //   - CPU 构建（默认 feature=cpu）：该层数仅作记录，llama.cpp 不会卸载到 GPU。
        //   - CUDA 构建（--features cuda）：按层数卸载到显存。
        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        // 加载 GGUF（文件损坏 / GGUF 版本不兼容 / 显存不足 / 内存不足 均在此出错）。
        // llama.cpp 对不认识的 general.architecture 只会返回 null，这里补充可读提示。
        Ok(LlamaModel::load_from_file(
            backend,
            &model_path,
            &params,
        )
        .map_err(|e| {
            format!(
                "模型加载失败: {}（若为 null result from llama cpp，通常是该 GGUF 的架构不受当前 llama.cpp 版本支持，或文件不完整/内存不足）",
                e
            )
        })?)
    })
    .await
    .map_err(|e| format!("加载线程异常: {}", e))?;

    // 6. 写入全局单例与标记。
    *state.llama_instance.lock().unwrap() = Some(loaded?);
    *state.current_loaded_model.lock().unwrap() = Some(item.clone());

    // 7. 同步清单中该模型的 loaded 状态。
    if let Ok(mut list) = fs_service::read_model_list(&base_dir) {
        if let Some(m) = list.iter_mut().find(|m| m.file_name == item.file_name) {
            m.loaded = true;
        }
        let _ = fs_service::write_model_list(&base_dir, &list);
    }

    log::info!("模型已加载: {} (n_gpu_layers={})", item.file_name, n_gpu_layers);
    Ok(item)
}

/// 卸载当前模型（阶段3：真实 llama 实例释放）。
/// 销毁 llama.cpp 上下文资源由 LlamaModel drop 自动完成；这里从全局单例取出后，
/// 在阻塞线程 drop，避免重量级析构阻塞 UI。
#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>) -> Result<(), String> {
    // 取出实例（置 None，触发所有权转移）。
    let old = state.llama_instance.lock().unwrap().take();
    if old.is_some() {
        // 在阻塞线程完成 drop，释放 llama.cpp 资源。
        tokio::task::spawn_blocking(move || drop(old))
            .await
            .map_err(|e| format!("卸载线程异常: {}", e))?;
    }

    // 清空加载标记，并同步清单中的 loaded 状态。
    let mut loaded = state.current_loaded_model.lock().unwrap();
    if let Some(item) = loaded.as_ref() {
        if let Ok(mut list) = fs_service::read_model_list(&state.base_dir) {
            if let Some(m) = list.iter_mut().find(|m| m.file_name == item.file_name) {
                m.loaded = false;
            }
            let _ = fs_service::write_model_list(&state.base_dir, &list);
        }
    }
    *loaded = None;

    log::info!("模型已卸载（llama 实例已释放）");
    Ok(())
}

/// 删除选中模型：若已加载则禁止删除；否则删文件 + 移除清单。
#[tauri::command]
pub fn delete_model(state: State<'_, AppState>, path: String) -> Result<(), String> {
    let mut list = fs_service::read_model_list(&state.base_dir)?;
    let item = list
        .iter()
        .find(|m| m.path == path || m.file_name == path)
        .ok_or_else(|| "模型未在清单中找到".to_string())?;

    // 检查是否已加载：加载中禁止删除。
    let loaded = state.current_loaded_model.lock().unwrap();
    if let Some(cur) = loaded.as_ref() {
        if cur.file_name == item.file_name {
            return Err("该模型正在加载，请先卸载后再删除".to_string());
        }
    }
    drop(loaded);

    // 提前取出要删除的文件名，避免借用冲突。
    let target_name = item.file_name.clone();

    // 删除物理文件：失败必须上报。
    // 若吞掉错误，清单已移除但文件仍在，再次导入会命中“模型文件已存在”而无法恢复。
    let file_path = state.base_dir.join("models").join(&target_name);
    if file_path.exists() {
        fs_service::delete_file(&file_path)?;
    }

    // 从清单移除并保存。
    list.retain(|m| m.file_name != target_name);
    fs_service::write_model_list(&state.base_dir, &list)?;
    log::info!("模型已删除: {}", target_name);
    Ok(())
}

/// 发送聊天（阶段4：真实流式推理）。
/// 入参：
///   - `history`：完整对话历史（含本轮用户消息）
///   - `params`：本轮推理参数（与前端表单字段对齐）
/// 事件：统一通过 `chat_token` 事件推送 ChatTokenEvent：
///   - kind="token"：增量文本
///   - kind="end"  ：生成结束，content 为完整回复
///   - kind="error"：生成异常，content 为错误信息
/// 返回：命令本身成功/失败（失败信息同时通过 error 事件推送，便于前端弹窗）。
#[tauri::command]
pub async fn send_chat(
    app: AppHandle,
    state: State<'_, AppState>,
    history: Vec<ChatMessage>,
    params: InferenceParams,
) -> Result<(), String> {
    if history.is_empty() {
        return Err("对话历史为空".to_string());
    }
    // ① 前置校验：必须已加载模型（未加载直接拒绝，不进入阻塞线程）。
    let model_loaded = {
        let guard = state
            .llama_instance
            .lock()
            .map_err(|_| "模型锁已损坏".to_string())?;
        guard.is_some()
    };
    if !model_loaded {
        return Err("尚未加载模型，请先加载模型".to_string());
    }
    // 并发保护：同一时间只允许一个生成任务。
    if state
        .is_generating
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("正在生成中，请稍候或先停止".to_string());
    }
    // 新一轮生成开始前清除终止标志。
    state.generation_stop_flag.store(false, Ordering::SeqCst);

    // 复制所需句柄后进入阻塞线程（推理是重 CPU/GPU 操作，严禁占用 Tauri 主线程）。
    let model_slot = Arc::clone(&state.llama_instance);
    let stop_flag = Arc::clone(&state.generation_stop_flag);
    let worker_app = app.clone();

    let joined = tokio::task::spawn_blocking(move || {
        run_generation(&worker_app, &model_slot, &history, &params, &stop_flag)
    })
    .await;

    // 无论成功失败都释放生成状态。
    state.is_generating.store(false, Ordering::SeqCst);

    let (event, result) = match joined {
        // 生成正常结束（含用户中途停止，已输出内容保留）。
        Ok(Ok(full_text)) => (
            ChatTokenEvent {
                kind: "end".to_string(),
                content: full_text,
            },
            Ok(()),
        ),
        // 推理内部错误：结构化错误同时走事件与返回值。
        Ok(Err(err)) => (
            ChatTokenEvent {
                kind: "error".to_string(),
                content: err.clone(),
            },
            Err(err),
        ),
        // 阻塞线程异常（panic 等）。
        Err(join_err) => {
            let err = format!("生成线程异常: {}", join_err);
            (
                ChatTokenEvent {
                    kind: "error".to_string(),
                    content: err.clone(),
                },
                Err(err),
            )
        }
    };
    let _ = app.emit("chat_token", event);
    result
}

/// 停止当前生成。
/// 只置位终止标志，采样循环下一次迭代检测到后自行退出；多次调用无害。
#[tauri::command]
pub async fn stop_generation(state: State<'_, AppState>) -> Result<(), String> {
    state.generation_stop_flag.store(true, Ordering::SeqCst);
    log::info!("收到停止生成请求");
    Ok(())
}

/// 阶段4：采样主循环（运行在阻塞线程）。
/// 流程：取模型 -> 组装 ChatML prompt -> 分词 -> 建上下文 -> prompt 解码 -> 逐 token 采样并推送。
/// 注意：整个生成期间持有模型锁，保证 llama 实例唯一。
fn run_generation(
    app: &AppHandle,
    model_slot: &Mutex<Option<LlamaModel>>,
    history: &[ChatMessage],
    params: &InferenceParams,
    stop_flag: &AtomicBool,
) -> Result<String, String> {
    // ---- ① 取已加载模型（未加载直接报错） ----
    let guard = model_slot.lock().map_err(|_| "模型锁已损坏".to_string())?;
    let model = guard
        .as_ref()
        .ok_or_else(|| "尚未加载模型，请先加载模型".to_string())?;

    // ---- ② 组装 ChatML 上下文 ----
    let prompt = build_chatml_prompt(history);
    // add_special=true 补 BOS，parse_special=true 解析 <|im_start|> 等特殊标记。
    let mut tokens = model.vocab().tokenize(prompt.as_bytes(), true, true);
    // 关键修复：ChatML 模板以 `{{- bos_token }}`（即 <s>）开头，
    // 但部分 GGUF 元数据为 add_bos_token=false，此时 llama.cpp 的 add_special
    // 不会真正补 BOS，模型收到缺少 <s> 的 prompt 会退化成空格/换行或立即 EOG（表现为“乱码”）。
    // 因此这里显式补齐 BOS，兼容 add_bos_token 为 true / false 的两类模型。
    let bos = model.vocab().bos();
    if tokens.first() != Some(&bos) {
        tokens.insert(0, bos);
    }
    if tokens.is_empty() {
        return Err("提示词分词结果为空".to_string());
    }

    let n_ctx = params.n_ctx.max(256);
    if tokens.len() >= n_ctx as usize {
        return Err(format!(
            "上下文不足：提示词 {} tokens 已超过 n_ctx={}，请减少对话历史或调大上下文",
            tokens.len(),
            n_ctx
        ));
    }

    // ---- ③ 创建上下文并解码整段 prompt ----
    let backend = backend()?;
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("创建推理上下文失败: {}", e))?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    batch
        .add_sequence(&tokens, 0, false)
        .map_err(|e| format!("填充 prompt 批次失败: {}", e))?;
    ctx.decode(&mut batch)
        .map_err(|e| format!("prompt 解码失败: {}", e))?;

    // ---- ④ 构造采样器：温度<=0 时贪心，否则 temp + top_p + dist ----
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(42);
    let mut sampler = if params.temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_p(params.top_p.clamp(0.0, 1.0), 1),
            LlamaSampler::dist(seed),
        ])
    };

    // ---- ⑤ 采样循环 ----
    let vocab = model.vocab();
    // raw：本次生成的完整原始文本（用于检测 ChatML 结束标记）。
    let mut raw = String::new();
    // emitted：已通过事件推送给前端的字节数（raw[..emitted] 即前端已显示内容）。
    let mut emitted = 0usize;
    // pending：增量 UTF-8 解码缓冲（一个多字节字符可能被拆到多个 token）。
    let mut pending: Vec<u8> = Vec::new();
    let mut n_pos = tokens.len() as i32;
    const END_MARKER: &str = "<|im_end|>";

    for _ in 0..params.max_new_tokens {
        // 停止标记判断：用户点击停止后，保留已输出内容并结束。
        if stop_flag.load(Ordering::SeqCst) {
            log::info!("生成被用户停止");
            break;
        }
        if n_pos >= n_ctx as i32 {
            log::warn!("达到上下文上限 n_ctx={}", n_ctx);
            break;
        }

        // 采样下一个 token
        let token = sampler.sample(&ctx, -1);
        // 停止符（EOS 等）直接结束
        if vocab.is_eog(token) {
            break;
        }
        sampler.accept(token);

        // 增量解码 token 字节：只取当前能构成合法 UTF-8 的前缀。
        pending.extend_from_slice(&vocab.token_to_piece(token, false, None));
        let valid_len = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(err) => err.valid_up_to(),
        };
        if valid_len > 0 {
            let text = String::from_utf8_lossy(&pending[..valid_len]).to_string();
            pending.drain(..valid_len);
            raw.push_str(&text);
        }

        // ChatML 结束标记截断：出现 <|im_end|> 时只输出其前面的内容。
        if let Some(pos) = raw.find(END_MARKER) {
            raw.truncate(pos);
            if raw.len() > emitted {
                let delta = raw[emitted..].to_string();
                emitted = raw.len();
                emit_token(app, &delta)?;
            }
            break;
        }

        // 事件推送：保留 marker 长度-1 的尾巴，避免把跨 token 的标记片段提前输出。
        let mut safe_end = raw.len().saturating_sub(END_MARKER.len() - 1);
        while safe_end > 0 && !raw.is_char_boundary(safe_end) {
            safe_end -= 1;
        }
        if safe_end > emitted {
            let delta = raw[emitted..safe_end].to_string();
            emitted = safe_end;
            emit_token(app, &delta)?;
        }

        // 把新 token 送回上下文解码，为下一次采样准备 logits。
        batch.clear();
        batch
            .add(token, n_pos, &[0], true)
            .map_err(|e| format!("填充生成批次失败: {}", e))?;
        n_pos += 1;
        ctx.decode(&mut batch)
            .map_err(|e| format!("生成解码失败: {}", e))?;
    }

    // ---- ⑥ 收尾：输出剩余合法内容，返回完整回复 ----
    let mut end = raw.len();
    while end > emitted && !raw.is_char_boundary(end) {
        end -= 1;
    }
    if end > emitted {
        let delta = raw[emitted..end].to_string();
        emitted = end;
        emit_token(app, &delta)?;
    }
    let full_text = raw[..emitted].to_string();
    log::info!("生成完成，共 {} 字符", full_text.chars().count());
    Ok(full_text)
}

/// 通过 chat_token 事件推送一个增量 token。
fn emit_token(app: &AppHandle, delta: &str) -> Result<(), String> {
    app.emit(
        "chat_token",
        ChatTokenEvent {
            kind: "token".to_string(),
            content: delta.to_string(),
        },
    )
    .map_err(|e| format!("推送 token 事件失败: {}", e))
}

/// 组装 ChatML 格式 prompt：
/// ```text
/// <|im_start|>user
/// 你好<|im_end|>
/// <|im_start|>assistant
/// ```
/// 说明：ChatML 以换行分隔 role 与正文；若目标模型要求特殊分隔符，
/// 可在此集中调整，不影响采样流程。
fn build_chatml_prompt(history: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in history {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("<|im_end|>\n");
    }
    // 预留 assistant 起始标记，模型从这里续写。
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ===== 阶段5：会话持久化 =====

/// 当前 unix 毫秒时间戳。
fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// 列出全部会话，按 updated_at 倒序。
#[tauri::command]
pub fn list_sessions(state: State<'_, AppState>) -> Result<Vec<Session>, String> {
    fs_service::read_sessions(&state.base_dir)
}

/// 新建会话：生成 UUID，默认标题“新会话”，空消息数组。
#[tauri::command]
pub fn create_session(state: State<'_, AppState>) -> Result<Session, String> {
    let now = now_millis();
    let session = Session {
        id: uuid::Uuid::new_v4().to_string(),
        title: "新会话".to_string(),
        created_at: now,
        updated_at: now,
        messages: vec![],
    };
    fs_service::write_session(&state.base_dir, &session)?;
    log::info!("新建会话: {}", session.id);
    Ok(session)
}

/// 覆盖保存会话（前端在生成结束时组装完整 Session 后调用）。
#[tauri::command]
pub fn save_session(state: State<'_, AppState>, session: Session) -> Result<(), String> {
    fs_service::write_session(&state.base_dir, &session)
}

/// 读取指定会话。
#[tauri::command]
pub fn load_session(state: State<'_, AppState>, session_id: String) -> Result<Session, String> {
    fs_service::read_session(&state.base_dir, &session_id)
}

/// 删除指定会话文件。
#[tauri::command]
pub fn delete_session(state: State<'_, AppState>, session_id: String) -> Result<(), String> {
    fs_service::delete_session(&state.base_dir, &session_id)
}

/// 读取推理配置。
#[tauri::command]
pub fn get_config(state: State<'_, AppState>) -> Result<AppConfig, String> {
    Ok(state.current_config.lock().unwrap().clone())
}

/// 保存推理配置。
#[tauri::command]
pub fn save_config(state: State<'_, AppState>, config: AppConfig) -> Result<(), String> {
    *state.current_config.lock().unwrap() = config.clone();
    fs_service::save_config(&state.base_dir, &config)
}
