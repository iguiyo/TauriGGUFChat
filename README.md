# TauriGGUFChat

一个基于 Tauri v2 + llama.cpp 的 Windows 本地 GGUF 聊天客户端。单文件程序、纯本地推理，不需要联网，也不依赖 Python / 显卡驱动。

## 功能

- 导入本地 GGUF 模型（导入即复制进软件内部模型库，源文件可自行删除）
- CPU 推理 + 流式输出，可随时停止生成
- 多会话管理与自动持久化（切换、删除、按更新时间排序）
- 推理参数可视化调整（上下文长度、温度、top_p、最大生成长度等）
- 便携模式：数据保存在程序同目录，拷贝整个文件夹即可迁移

## 使用

1. 下载 `TauriGGUFChat-v0.1.1-windows-x64.exe`，放到任意一个有写权限的文件夹
2. 双击运行（首次启动会在 exe 旁生成 `data\` 目录）
3. 点击「导入 GGUF」，选择本地模型文件；程序会复制一份到 `data\models\`
4. 在下拉框选中模型 → 「加载选中模型」
5. 在右侧输入框输入内容开始对话

### 数据存放位置

便携模式，数据就在 exe 同级的 `data\` 目录：

```
data\
├── models\          # 导入的模型副本 + model_list.json
├── sessions\        # 会话记录
├── chat_history\    # 历史目录
└── config.json      # 推理参数
```

如果 exe 所在目录不可写（例如放进 `Program Files`），程序会自动回退到 `%APPDATA%\com.tauriggufchat.app`。

## 系统要求

- Windows 10 1809+ / Windows 11，x64
- WebView2 运行时（Win11 与较新 Win10 自带；老系统需安装 Evergreen 运行时）
- 内存需大于模型体积，建议 8 GB 以上

## 从源码构建

前置：Rust 工具链、CMake、Visual Studio Build Tools（C++ 桌面开发）。

```bash
cd src-tauri
cargo build --release
```

产物：`src-tauri/target/release/tauri-gguf-chat.exe`。

`llama-cpp-2` 依赖本地 vendored 源码 `third_party/llama-cpp-rs`（含上游 `main` 分支的 llama.cpp 子模块，用于支持 `lfm3` 等较新架构）。该目录未纳入仓库，克隆后需单独拉取：

```bash
git clone --depth 1 --branch main https://github.com/utilityai/llama-cpp-rs third_party/llama-cpp-rs
cd third_party/llama-cpp-rs
git submodule update --init --depth 1 llama-cpp-sys-2/llama.cpp
```

### 可选：CUDA 构建

```bash
cargo tauri build --features cuda
```

## 技术栈

- 前端：原生 HTML / CSS / JavaScript（构建时内嵌进二进制）
- 后端：Rust + Tauri v2
- 推理：[llama-cpp-2](https://github.com/utilityai/llama-cpp-rs)（llama.cpp 绑定，默认 CPU 后端）

## 说明

- 本程序不做任何网络请求，模型与对话记录全部留在本地
- 未做代码签名，首次运行可能被 SmartScreen 拦截，选择「更多信息 → 仍要运行」即可

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。

项目中使用的第三方组件（llama.cpp、llama-cpp-rs、Tauri 等）遵循其各自的许可证。
