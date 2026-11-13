//! 公共数据结构定义。
//! 前端与 Rust 后端（通过 tauri::command + serde）共享这些结构。

use serde::{Deserialize, Serialize};

/// 模型清单文件（models/model_list.json）的顶层结构。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListFile {
    /// 清单格式版本
    pub version: u32,
    /// 全部模型记录
    pub models: Vec<ModelItem>,
}

impl Default for ModelListFile {
    fn default() -> Self {
        Self {
            version: 1,
            models: vec![],
        }
    }
}

/// 一条模型记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelItem {
    /// 显示名称（用户可自定义）
    pub name: String,
    /// GGUF 文件名（models 目录内）
    pub file_name: String,
    /// 完整文件路径
    pub path: String,
    /// 模型大小（字节）
    pub size: u64,
    /// 是否已加载
    pub loaded: bool,
}

/// 全局推理参数配置，持久化到 config.json。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// 上下文字符数
    pub n_ctx: u32,
    /// GPU 卸载层数（默认 12，适配 4G 显存）
    pub n_gpu_layers: u32,
    /// 采样温度
    pub temperature: f32,
    /// 最大新生成 token 数
    pub max_new_tokens: u32,
    /// top-p 采样
    pub top_p: f32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            n_ctx: 16384,
            n_gpu_layers: 12,
            temperature: 0.7,
            max_new_tokens: 1024,
            top_p: 0.95,
        }
    }
}

/// 聊天消息，一条对话记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// 消息角色：user / assistant
    pub role: String,
    /// 消息内容
    pub content: String,
}

/// 阶段5：会话元信息。
/// 持久化到 app_data_dir/sessions/{id}.json，id 为 UUID v4。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// 会话唯一 id（UUID v4，同时作为文件名）
    pub id: String,
    /// 会话标题（默认“新会话”，可用首条用户消息前 20 字符作为标题）
    pub title: String,
    /// 创建时间（unix 毫秒）
    pub created_at: i64,
    /// 最后更新时间（unix 毫秒）
    pub updated_at: i64,
    /// 会话内的完整消息列表
    pub messages: Vec<ChatMessage>,
}

/// 阶段4：流式事件负载（事件名 chat_token）。
/// kind 取值：
///   - "token"：一个文本片段（增量）
///   - "end"：生成结束，content 为完整回复
///   - "error"：生成异常，content 为结构化错误信息
#[derive(Debug, Clone, Serialize)]
pub struct ChatTokenEvent {
    /// 事件类型：token / end / error
    pub kind: String,
    /// 事件内容
    pub content: String,
}

/// 阶段3：推理参数。
/// 与前端表单字段一一对齐，供后续构造 llama-cpp-2 的模型/上下文参数使用。
/// 其中 `n_gpu_layers` 直接用于 `LlamaModelParams`；
/// `n_ctx`/`temperature`/`max_new_tokens`/`top_p` 预留用于阶段4的上下文与采样。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParams {
    /// 上下文 token 数（构造 LlamaContextParams 使用）
    pub n_ctx: u32,
    /// GPU 卸载层数（构造 LlamaModelParams 使用；CPU 构建时无效）
    pub n_gpu_layers: u32,
    /// 采样温度
    pub temperature: f32,
    /// 最大新生成 token 数
    pub max_new_tokens: u32,
    /// top-p 采样
    pub top_p: f32,
}

impl From<AppConfig> for InferenceParams {
    fn from(cfg: AppConfig) -> Self {
        Self {
            n_ctx: cfg.n_ctx,
            n_gpu_layers: cfg.n_gpu_layers,
            temperature: cfg.temperature,
            max_new_tokens: cfg.max_new_tokens,
            top_p: cfg.top_p,
        }
    }
}

impl From<InferenceParams> for AppConfig {
    fn from(p: InferenceParams) -> Self {
        Self {
            n_ctx: p.n_ctx,
            n_gpu_layers: p.n_gpu_layers,
            temperature: p.temperature,
            max_new_tokens: p.max_new_tokens,
            top_p: p.top_p,
        }
    }
}