//! 文件系统服务：JSON 读写与基础文件工具。
//! 所有路径基于传入的 base_dir（由 tauri path resolver 提供）。
//! 供模型管理 / 对话历史等阶段复用。

use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::types::Session;

/// 读取 JSON 文件并反序列化为指定类型。
pub fn read_json<T>(path: &Path) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    let content = fs::read_to_string(path).map_err(|e| format!("读取文件失败: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("JSON 解析失败: {}", e))
}

/// 将指定类型序列化为 JSON 并写入文件（自动创建父目录）。
pub fn write_json<T>(path: &Path, value: &T) -> Result<(), String>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建目录失败: {}", e))?;
    }
    let content =
        serde_json::to_string_pretty(value).map_err(|e| format!("JSON 序列化失败: {}", e))?;
    fs::write(path, content).map_err(|e| format!("写入文件失败: {}", e))
}

/// 读取 model_list.json，返回模型数组。
/// 文件不存在时返回空列表。
pub fn read_model_list(base_dir: &Path) -> Result<Vec<crate::types::ModelItem>, String> {
    let path = base_dir.join("models").join("model_list.json");
    if !path.exists() {
        return Ok(vec![]);
    }
    let list: crate::types::ModelListFile =
        read_json(&path).unwrap_or(crate::types::ModelListFile::default());
    Ok(list.models)
}

/// 将模型数组写回 model_list.json。
pub fn write_model_list(base_dir: &Path, models: &[crate::types::ModelItem]) -> Result<(), String> {
    let path = base_dir.join("models").join("model_list.json");
    let file = crate::types::ModelListFile {
        version: 1,
        models: models.to_vec(),
    };
    write_json(&path, &file)
}

/// 读取 config.json，构造默认值。
pub fn load_config(base_dir: &Path) -> crate::types::AppConfig {
    let path = base_dir.join("config.json");
    read_json::<crate::types::AppConfig>(&path).unwrap_or_default()
}

/// 保存 config.json。
pub fn save_config(base_dir: &Path, config: &crate::types::AppConfig) -> Result<(), String> {
    let path = base_dir.join("config.json");
    write_json(&path, config)
}

/// 复制文件到目标路径（自动创建父目录）。
/// 若目标已存在则返回错误（避免误覆盖）。
pub fn copy_file(src: &Path, dst: &Path) -> Result<(), String> {
    if dst.exists() {
        return Err(format!("目标文件已存在: {}", dst.display()));
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建目录失败: {}", e))?;
    }
    fs::copy(src, dst).map_err(|e| format!("复制文件失败: {}", e))?;
    Ok(())
}

/// 删除文件。
pub fn delete_file(path: &Path) -> Result<(), String> {
    fs::remove_file(path).map_err(|e| format!("删除文件失败: {}", e))
}

/// 返回不存在冲突的模型列表（按文件名查重）。
/// 若存在同名文件返回错误。
pub fn ensure_unique_filename(base_dir: &Path, file_name: &str) -> Result<(), String> {
    let target = base_dir.join("models").join(file_name);
    if target.exists() {
        return Err(format!("模型文件已存在: {}", file_name));
    }
    Ok(())
}

/// 计算文件大小（字节）。
pub fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// 从路径提取文件名。
pub fn file_name_from(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ===== 阶段5：会话文件读写 =====

/// 会话目录：app_data_dir/sessions。
pub fn sessions_dir(base_dir: &Path) -> PathBuf {
    base_dir.join("sessions")
}

/// 确保会话目录存在（不存在则自动创建）。
pub fn ensure_sessions_dir(base_dir: &Path) -> Result<PathBuf, String> {
    let dir = sessions_dir(base_dir);
    fs::create_dir_all(&dir).map_err(|e| format!("创建 sessions 目录失败: {}", e))?;
    Ok(dir)
}

/// 校验会话 id：仅允许小写字母、数字与连字符，防御路径穿越（禁止 `/`、`.`、`..` 等）。
fn validate_session_id(id: &str) -> Result<(), String> {
    let valid = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if valid {
        Ok(())
    } else {
        Err(format!("非法会话 id: {}", id))
    }
}

/// 会话文件路径：sessions/{id}.json。
pub fn session_path(base_dir: &Path, id: &str) -> Result<PathBuf, String> {
    validate_session_id(id)?;
    Ok(sessions_dir(base_dir).join(format!("{}.json", id)))
}

/// 读取单个会话文件。
pub fn read_session(base_dir: &Path, id: &str) -> Result<Session, String> {
    let path = session_path(base_dir, id)?;
    if !path.exists() {
        return Err(format!("会话不存在: {}", id));
    }
    read_json(&path)
}

/// 覆盖写入会话文件（自动创建 sessions 目录）。
pub fn write_session(base_dir: &Path, session: &Session) -> Result<(), String> {
    ensure_sessions_dir(base_dir)?;
    let path = session_path(base_dir, &session.id)?;
    write_json(&path, session)
}

/// 读取全部会话，按 updated_at 倒序返回；损坏文件仅告警跳过，不影响整体列表。
pub fn read_sessions(base_dir: &Path) -> Result<Vec<Session>, String> {
    let dir = ensure_sessions_dir(base_dir)?;
    let entries = fs::read_dir(&dir).map_err(|e| format!("读取 sessions 目录失败: {}", e))?;
    let mut sessions = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("读取会话目录项失败: {}", e))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match read_json::<Session>(&path) {
            Ok(session) => sessions.push(session),
            Err(e) => log::warn!("跳过无法解析的会话文件 {:?}: {}", path, e),
        }
    }
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(sessions)
}

/// 删除会话文件（文件不存在视为成功）。
pub fn delete_session(base_dir: &Path, id: &str) -> Result<(), String> {
    let path = session_path(base_dir, id)?;
    if !path.exists() {
        return Ok(());
    }
    fs::remove_file(&path).map_err(|e| format!("删除会话文件失败: {}", e))
}