# 课堂知识伴侣

[中文](README_ZH.md) | [English](README.md)

> 上传课堂视频，一键获取可检索、可问答、支持多语言的学习笔记。

## 项目亮点

- **端到端流水线**：Faster-Whisper 转写 → LLM 知识点梳理 → 幻灯片抽取 & PDF 合成 → 可选多语言翻译。
- **多讲座工作区**：在界面中上传、监控、删除讲座，每个讲座有独立的转录与幻灯片。
- **多语种版本**：界面可切换中英，并可在生成讲座时选择任意语言（中文、英文、日语、韩语、法语等）输出对应的转录与梳理。
- **双通道问答**：一次调用扫描整篇转录（定位时间戳 + 小结），另一次调用让大模型自由多轮作答，两个结果并列展示，方便对比课堂内容与模型推理。
- **可视化配置**：在“系统设置”面板中即可修改 LLM 接口、Whisper 模型以及讲座存储目录，保存后自动写入 `settings.json` 并重启应用。
- **PDF 幻灯片对齐**：上传教师的 PDF 幻灯片后，系统会每隔 2 秒采样视频帧，用 ResNet 特征做相似度匹配，按页顺序把幻灯片与知识点时间段绑定。
- **部署友好**：所有产出放在 `data/lectures/<lecture_id>/`，易于备份或迁移。

## 快速上手

### 1. 安装依赖

```bash
git clone https://github.com/1368353446/ClassAtlas.git
cd ClassAtlas
pip install -r requirements.txt
```

### 2. 配置 `settings.json`

首次启动时会自动生成 `settings.json`（已加入 .gitignore）。可直接编辑该文件，也可在界面里的“系统设置”面板修改。示例：

```json
{
  "llm_model": "Qwen/Qwen3-8B",
  "llm_base_url": "https://api-inference.modelscope.cn/v1/",
  "llm_api_key": "<你的 ModelScope Key>",
  "whisper_model": "medium",
  "lecture_root": "data/lectures"
}
```

### 3. 处理讲座

CLI 示例：

```bash
python process_lecture.py \
  --video path/to/video.mp4 \
  --output data/lectures/lecture-a \
  --lecture-id lecture-a \
  --title "Lecture A"
```

在 UI 中上传视频也会触发同样的流程，并可附加 PDF 幻灯片；系统会自动匹配每一页与对应的课堂时刻，后续在知识点里展示。

### 4. 启动界面

```bash
streamlit run streamlit_app.py
```

使用提示：

1. 右上角切换 UI 语言（中文/English）。
2. “讲座管理” 上传/删除并查看处理进度。
3. “课堂知识浏览” 选择讲座与版本，查看视频、总结、知识点、幻灯片、完整转录。
4. 打开侧边栏“知识问答助手”，即可同时获得“转录定位+总结”与“大模型多轮对话”两类回答，系统会自动匹配提问语言。
5. 在“系统设置”区域可调整 LLM 接口、Whisper 模型以及讲座存储目录；保存后应用会自动刷新。多段 LLM 处理会自动覆盖整段转录（转折点 → 分段 → 详尽总结），并在知识点中展示详细内容和对应 PDF 页码。

## 目录概览

- `streamlit_app.py`：主界面（管理 / 浏览 / 问答）。
- `process_lecture.py`：CLI 流水线入口。
- `app/config.py`：环境与路径配置。
- `app/pipeline/`：转写、LLM 梳理、幻灯片提取及调度。
- `app/translation.py`：多语言翻译，结果写入 `translations/<lang>/`。
- `app/lectures.py`：讲座元数据（源语言、翻译列表、时间戳）。
- `app/knowledge_base.py`、`app/loaders.py`、`app/models.py`：数据结构与读取。
- `app/qa.py`：同时提供“转录定位”与“多轮自由问答”两套链路，统一走同一个大模型。

## 部署建议

- 将 `data/lectures/` 映射到持久化存储或对象存储。
- 不要提交 `settings.json`（已在 `.gitignore` 中）。
- 依赖集中在 `requirements.txt`，可按部署环境（如 GPU Whisper）自行增删。
- 通过 cron/CI 运行 `process_lecture.py` 可批量处理；前端可部署在 Streamlit Cloud、Docker 或任何支持 Python 的服务器。
- 若目录中已存在 `transcript_segments.json`，重新处理讲座时会自动复用该字幕，避免重复跑 Whisper。

## 后续方向

- 接入更多资料源（课件、讲义、练习题）增强检索上下文。
- 将知识点编辑功能接入数据库，支持多人协作。
- 为 Streamlit 界面加入登录/权限控制，方便课堂或团队共享。
