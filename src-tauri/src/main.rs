//! Tauri 应用入口。
//! 注册所有 tauri command、初始化共享状态并启动窗口。

// 打包为 Windows 桌面应用时隐藏控制台窗口（仅 release 构建生效）。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod api_server;
mod commands;
mod fs_service;
mod types;

use commands::AppState;
use tauri::Manager;

/// 解析数据目录（便携模式）：固定为 exe 同级目录下的 `data` 文件夹，
/// 模型 / 会话 / 配置都写在这里，拷贝整个程序文件夹即可完整迁移。
/// 若该位置不可写（例如程序被放进 Program Files，普通权限无法创建目录），
/// 回退到系统应用数据目录，保证程序仍能启动。
fn resolve_data_dir(app: &tauri::App) -> std::path::PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let portable = parent.join("data");
            if std::fs::create_dir_all(&portable).is_ok() {
                return portable;
            }
        }
    }
    app.path()
        .app_data_dir()
        .expect("无法解析应用数据目录")
}

fn main() {
    env_logger::init();
    log::info!("TauriGGUFChat starting (stage 5)");

    tauri::Builder::default()
        // 单实例插件必须最先注册：第二个进程启动时直接唤起已有窗口并退出。
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        // 文件选择对话框插件。
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // 便携模式：数据目录取 exe 同级的 data 目录，并在其中补齐目录结构。
            let data_dir = resolve_data_dir(app);
            std::fs::create_dir_all(data_dir.join("models"))
                .expect("无法创建 models 目录");
            std::fs::create_dir_all(data_dir.join("chat_history"))
                .expect("无法创建 chat_history 目录");
            // 阶段5：会话持久化目录。
            std::fs::create_dir_all(data_dir.join("sessions"))
                .expect("无法创建 sessions 目录");

            // 首次运行生成默认 config.json。
            let config_path = data_dir.join("config.json");
            if !config_path.exists() {
                let _ = fs_service::write_json(
                    &config_path,
                    &crate::types::AppConfig::default(),
                );
            }
            // 首次运行生成默认 model_list.json。
            let list_path = data_dir.join("models").join("model_list.json");
            if !list_path.exists() {
                let _ = fs_service::write_json(
                    &list_path,
                    &crate::types::ModelListFile::default(),
                );
            }

            // 注入共享状态。
            app.manage(AppState::new(data_dir.clone()));

            // 本地 API 控制状态；默认随程序启动开启（可在界面里关闭/再开启）。
            let api_control = api_server::ApiControl::new();
            if let Err(e) = api_control.start(app.handle().clone()) {
                log::warn!("本地 API 启动失败: {}", e);
            }
            app.manage(api_control);

            log::info!("TauriGGUFChat initialized, base_dir = {:?}", data_dir);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::import_model,
            commands::refresh_model_list,
            commands::load_model,
            commands::unload_model,
            commands::delete_model,
            commands::send_chat,
            commands::stop_generation,
            commands::get_config,
            commands::save_config,
            // 阶段5：会话持久化
            commands::list_sessions,
            commands::create_session,
            commands::save_session,
            commands::load_session,
            commands::delete_session,
            // 本地 API 开关
            api_server::api_status,
            api_server::start_api,
            api_server::stop_api,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}