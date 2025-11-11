# Lecture Knowledge Companion

[English](README.md) | [中文](README_ZH.md)

> Streamlit + LangChain toolkit that turns raw lecture videos into searchable, multilingual study companions.

## Highlights

- **End-to-end pipeline** – Faster-Whisper transcription, LLM-based knowledge outlining, slide extraction, PDF bundling, and optional multi-language translation.
- **Multi-lecture workspace** – Upload, monitor, and delete lectures directly in the UI; each run keeps its own transcripts, slides, and FAISS index.
- **Bilingual experience** – UI toggles between English and Chinese; you can auto-generate translated transcripts/outlines so every lecture exposes multiple language versions.
- **Retrieval-first QA** – Lightweight BM25 retriever plus optional FAISS vector store; you can fall back to citation-style answers when LLMs are disabled.
- **Collaboration-ready** – Assets live under `data/lectures/<lecture_id>/`, making it easy to sync or deploy in different environments.

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/1368353446/ClassAtlas.git
cd ClassAtlas
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and adjust values:

```env
LECTURE_ROOT=data/lectures
MODELSCOPE_API_KEY=sk-your-modelscope-key
LLM_MODEL_NAME=Qwen/Qwen3-8B
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

> Need FAISS? Install a build compatible with your NumPy version, e.g. `pip install "numpy<2" faiss-cpu`.

### 3. Process lectures

You can run the full pipeline via CLI or the UI:

```bash
python process_lecture.py \
  --video path/to/video.mp4 \
  --output data/lectures/lecture-a \
  --lecture-id lecture-a \
  --title "Lecture A"
```

Or upload a video from the “Lecture Management” panel; optionally select translation targets so the translated transcripts/outlines are generated right after the base run.

### 4. Launch the UI

```bash
streamlit run streamlit_app.py
```

Inside the app:

1. Use the top-right language toggle (English/中文).
2. “Lecture Management” uploads and deletes lectures, showing stage-by-stage timings.
3. “Lecture Explorer” lets you choose a lecture & version to review the video, overview, per-point insights, slides, and the entire transcript.
4. Open the sidebar “Knowledge Q&A” to ask questions while studying; BM25 works offline, FAISS enables semantic search, and you can switch off the LLM for citation-style answers.

## Project Layout

- `streamlit_app.py` – UI with management, explorer, and Q&A panels.
- `process_lecture.py` – CLI entry point mirroring the end-to-end pipeline.
- `app/config.py` – paths + environment management.
- `app/pipeline/` – transcription, LLM outlining, slide extraction, and coordination (`runner.py`).
- `app/translation.py` – optional translation workflow (stores outputs under `translations/<lang>/`).
- `app/lectures.py` – metadata store (source language, translations, timestamps).
- `app/knowledge_base.py`, `app/loaders.py`, `app/models.py` – data access helpers.
- `app/retrievers.py`, `app/vector_store.py` – BM25 retriever and FAISS integration (with graceful fallback).
- `app/qa.py` – LangChain-based QA chain with LLM on/off switch.
- `app/editor.py` – placeholder for natural-language editing.

## Deployment Notes

- **Data directory** – By default everything is written to `data/lectures/`. Point `LECTURE_ROOT` to persistent storage in production.
- **Secrets** – Never commit `.env`; use `.env.example` as the template.
- **Dependencies** – Use `requirements.txt` to pin versions. If you rely on GPU FAISS, adjust the requirement accordingly.
- **Automation** – Schedule `process_lecture.py` for batch jobs; host the Streamlit app via Streamlit Cloud, containers, or any server with Python.

## Next Steps

- Plug in additional document sources (slides metadata, teacher notes) by adjusting `KnowledgeQASystem` inputs.
- Replace the placeholder editor with a formal revision workflow.
- Integrate authentication or classroom-level permissions if sharing across teams.

Looking for the Chinese guide? See [README_ZH.md](README_ZH.md).
