# Lecture Knowledge Companion

[English](README.md) | [中文](README_ZH.md)

> Streamlit + LangChain toolkit that turns raw lecture videos into searchable, multilingual study companions.

## Highlights

- **End-to-end pipeline** – Faster-Whisper transcription, LLM-based knowledge outlining, slide extraction, PDF bundling, and optional multi-language translation.
- **Multi-lecture workspace** – Upload, monitor, and delete lectures directly in the UI; each run keeps its own transcripts and slides.
- **Multilingual versions** – UI toggles between English and Chinese, while transcript/outline translations can target any configured language (Chinese, English, Japanese, Korean, French, etc.).
- **Dual QA modes** – One call scans the entire transcript (timestamps + summaries) while another runs a multi-turn free-form LLM chat; outputs are concatenated so users see grounded hits and open-ended answers side by side.
- **Configurable runtime** – Adjust LLM endpoint, Whisper size, and lecture storage directory directly inside the System Settings panel; changes persist to `settings.json` and trigger an automatic reload.
- **Slide alignment from PDF** – Upload the instructor’s slides as a PDF; each page is matched to video frames (ResNet features sampled every 2s) and attached to the relevant knowledge segments in chronological order.
- **Collaboration-ready** – Assets live under `data/lectures/<lecture_id>/`, making it easy to sync or deploy in different environments.

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/1368353446/ClassAtlas.git
cd ClassAtlas
pip install -r requirements.txt
```

### 2. Configure settings

On first launch the app creates `settings.json` (ignored by git). Edit it directly or use the in-app **System Settings** panel. Example structure:

```json
{
  "llm_model": "Qwen/Qwen3-8B",
  "llm_base_url": "https://api-inference.modelscope.cn/v1/",
  "llm_api_key": "<your ModelScope key>",
  "whisper_model": "medium",
  "lecture_root": "data/lectures"
}
```


### 3. Process lectures

You can run the full pipeline via CLI or the UI:

```bash
python process_lecture.py \
  --video path/to/video.mp4 \
  --output data/lectures/lecture-a \
  --lecture-id lecture-a \
  --title "Lecture A"
```

Or upload a video from the “Lecture Management” panel. You can optionally attach a slides PDF; each page is aligned to the video (sampled every 2 seconds) and later mapped onto the summarized knowledge segments.

### 4. Launch the UI

```bash
streamlit run streamlit_app.py
```

Inside the app:

1. Use the top-right language toggle (English/中文).
2. “Lecture Management” uploads and deletes lectures, showing stage-by-stage timings.
3. “Lecture Explorer” lets you choose a lecture & version to review the video, overview, per-point insights, slides, and the entire transcript.
4. Open the sidebar “Knowledge Q&A” to run transcript-grounded lookups (timestamps + summaries) and an optional multi-turn free-form chat in parallel; each call auto-detects the user language.

### 5. Tune runtime settings

- Visit the **System Settings** section (below Lecture Management) to edit:
  - LLM endpoint (model/base URL/API key)
  - Default Whisper model size
  - Lecture storage directory
- Saving triggers a quick reload so the new configuration is applied everywhere.
- Knowledge segments are now created via multi-step LLM analysis (turning points → block summaries → rich content). Every segment surfaces title, summary, detailed notes, teaching method, emphasis, timestamps, and any attached PDF pages.

## Project Layout

- `streamlit_app.py` – UI with management, explorer, and Q&A panels.
- `process_lecture.py` – CLI entry point mirroring the end-to-end pipeline.
- `app/config.py` – paths + environment management.
- `app/pipeline/` – transcription, LLM outlining, slide extraction, and coordination (`runner.py`).
- `app/translation.py` – optional translation workflow (stores outputs under `translations/<lang>/`).
- `app/lectures.py` – metadata store (source language, translations, timestamps).
- `app/knowledge_base.py`, `app/loaders.py`, `app/models.py` – data access helpers.
- `app/qa.py` – transcript-grounded QA plus a multi-turn general chat agent, both sharing the same LLM backend.
- `app/editor.py` – placeholder for natural-language editing.

## Deployment Notes

- **Data directory** – By default everything is written to `data/lectures/`. Point `LECTURE_ROOT` to persistent storage in production.
- **Secrets** – Never commit `settings.json`; it already lives in `.gitignore`.
- **Dependencies** – Use `requirements.txt` to pin versions; remove or add packages (e.g., GPU-accelerated Whisper) as needed for your stack.
- **Automation** – Schedule `process_lecture.py` for batch jobs; host the Streamlit app via Streamlit Cloud, containers, or any server with Python.
- **Caching** – If `transcript_segments.json` already exists for a lecture, the pipeline skips Whisper and reuses the cached transcript, saving time on re-runs.

## Next Steps

- Plug in additional document sources (slides metadata, teacher notes) by extending the transcript prompt before calling `TranscriptQASystem`.
- Replace the placeholder editor with a formal revision workflow.
- Integrate authentication or classroom-level permissions if sharing across teams.
