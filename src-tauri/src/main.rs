//! Tauri 应用入口。
//! 注册所有 tauri command、初始化共享状态并启动窗口。

// 打包为 Windows 桌面应用时隐藏控制台窗口（仅 release 构建生效）。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod fs_service;
mod types;

use commands::AppState;
use tauri::Manager;

fn main() {
    env_logger::init();
    log::info!("TauriGGUFChat starting (stage 5)");

    tauri::Builder::default()
        // 文件选择对话框插件。
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // 解析应用数据目录作为基础目录，并在其中补齐目录结构。
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("无法解析应用数据目录");
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
            app.manage(AppState::new(data_dir));

            log::info!(
                "TauriGGUFChat initialized, base_dir = {:?}",
                app.path().app_data_dir().unwrap()
            );
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}