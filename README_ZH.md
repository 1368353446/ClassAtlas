# 课堂知识伴侣

[中文](README_ZH.md) | [English](README.md)

> 上传课堂视频，一键获取可检索、可问答、支持多语言的学习笔记。

## 项目亮点

- **端到端流水线**：Faster-Whisper 转写 → LLM 知识点梳理 → 幻灯片抽取 & PDF 合成 → 可选多语言翻译。
- **多讲座工作区**：在界面中上传、监控、删除讲座，每个讲座有独立的转录、幻灯片和 FAISS 索引。
- **双语体验**：UI 可切换中英；生成讲座时还能顺带创建其他语言版本，方便跨语种复习。
- **检索优先 QA**：自带 BM25，可选 FAISS；可以关闭 LLM，只查看引用原文的回答。
- **部署友好**：所有产出放在 `data/lectures/<lecture_id>/`，易于备份或迁移。

## 快速上手

### 1. 安装依赖

```bash
git clone https://github.com/1368353446/ClassAtlas.git
cd ClassAtlas
pip install -r requirements.txt
```

### 2. 配置 `.env`

```env
LECTURE_ROOT=data/lectures
MODELSCOPE_API_KEY=sk-your-modelscope-key
LLM_MODEL_NAME=Qwen/Qwen3-8B
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

> 如需 FAISS，请确保 NumPy 版本匹配，例如 `pip install "numpy<2" faiss-cpu`。

### 3. 处理讲座

CLI 示例：

```bash
python process_lecture.py \
  --video path/to/video.mp4 \
  --output data/lectures/lecture-a \
  --lecture-id lecture-a \
  --title "Lecture A"
```

在 UI 中上传视频也会触发同样的流程，并可勾选额外语言进行翻译。

### 4. 启动界面

```bash
streamlit run streamlit_app.py
```

使用提示：

1. 右上角切换 UI 语言（中文/English）。
2. “讲座管理” 上传/删除并查看处理进度。
3. “课堂知识浏览” 选择讲座与版本，查看视频、总结、知识点、幻灯片、完整转录。
4. 打开侧边栏“知识问答助手”即可边看边问。

## 目录概览

- `streamlit_app.py`：主界面（管理 / 浏览 / 问答）。
- `process_lecture.py`：CLI 流水线入口。
- `app/config.py`：环境与路径配置。
- `app/pipeline/`：转写、LLM 梳理、幻灯片提取及调度。
- `app/translation.py`：多语言翻译，结果写入 `translations/<lang>/`。
- `app/lectures.py`：讲座元数据（源语言、翻译列表、时间戳）。
- `app/knowledge_base.py`、`app/loaders.py`、`app/models.py`：数据结构与读取。
- `app/retrievers.py`、`app/vector_store.py`：BM25 与 FAISS 检索。
- `app/qa.py`：LangChain 问答链，可配置是否调用 LLM。

## 部署建议

- 将 `data/lectures/` 映射到持久化存储或对象存储。
- 不要提交 `.env`，可使用 `.env.example` 做模板。
- 依赖集中在 `requirements.txt`，若使用 GPU FAISS 可自行修改。
- 通过 cron/CI 运行 `process_lecture.py` 可批量处理；前端可部署在 Streamlit Cloud、Docker 或任何支持 Python 的服务器。

英文说明请参阅 [README.md](README.md)。
