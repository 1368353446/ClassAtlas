from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import List
from uuid import uuid4

import streamlit as st

from app import build_assets
from app import config as app_config
from app.config import LECTURE_ROOT, UIConfig
from app.knowledge_base import slides_by_index
from app.lectures import (
    LectureRecord,
    compute_file_hash,
    get_lecture_dir,
    list_lectures,
    load_metadata,
    save_metadata,
    slugify,
    update_translations,
    METADATA_FILENAME,
)
from app.pipeline.runner import run_pipeline
from app.qa import GeneralChatAgent, TranscriptQASystem
from app.settings_manager import get_settings, update_settings
from app.translation import LANGUAGE_CONFIG, translate_outputs

STEP_META = {
    "transcription": {"label_key": "step_transcription", "metric_key": "segments", "metric_label_key": "metric_segments"},
    "outline": {"label_key": "step_outline", "metric_key": "knowledge_points", "metric_label_key": "metric_points"},
    "slides": {"label_key": "step_slides", "metric_key": "slides", "metric_label_key": "metric_slides"},
    "enrichment": {"label_key": "step_enrichment", "metric_key": None, "metric_label_key": ""},
}

UI_TEXT = {
    "zh": {
        "ui_language": "界面语言",
        "section_manage": "讲座管理",
        "input_title": "讲座名称",
        "upload_video": "上传课堂视频",
        "upload_pdf": "上传幻灯片 PDF（可选）",
        "translation_select": "选择需要同步生成的其他语言版本",
        "btn_start": "开始处理",
        "warn_fill": "请填写讲座名称并上传视频。",
        "status_processing": "正在处理讲座...",
        "status_complete": "处理完成",
        "summary_template": "讲座《{title}》处理完成：{segments} 条转录片段 · {points} 个知识点 · {slides} 张幻灯片。",
        "latest_progress": "最新处理进度",
        "lecture_list": "讲座列表",
        "no_lectures": "当前还没有任何讲座，上传视频即可自动创建。",
        "create_conflict": "已有同名讲座，请更换名称。",
        "delete": "删除",
        "delete_success": "已删除讲座《{title}》",
        "section_view": "课堂知识浏览",
        "select_lecture": "选择讲座",
        "select_language": "选择版本",
        "section_video": "课堂视频",
        "no_video": "未找到该讲座对应的视频文件。",
        "section_summary": "整体梳理",
        "session_topic": "主题：{topic}",
        "section_points": "知识点",
        "no_points": "暂无知识点数据。",
        "time_range": "时间段：{start:.2f}s - {end:.2f}s",
        "summary_label": "摘要：",
        "content_label": "详细梳理：",
        "teaching_label": "教学方法：",
        "emphasis_label": "强调：",
        "slides_linked": "关联幻灯片：",
        "section_slides": "幻灯片截图",
        "no_slides": "尚未提取幻灯片。",
        "section_transcript": "完整转录",
        "expand_transcript": "展开查看转录文本",
        "section_qa": "知识问答助手",
        "qa_enable": "启用问答助手",
        "qa_disabled": "勾选上方复选框以启用问答。",
        "qa_no_lectures": "暂无讲座可用。",
        "qa_select_lecture": "选择讲座",
        "qa_select_language": "选择版本",
        "qa_question": "问题",
        "qa_submit": "生成回答",
        "qa_recent": "最近的回答",
        "qa_feature_transcript": "使用课堂转录定位并总结",
        "qa_feature_general": "使用大模型自由作答（多轮）",
        "qa_transcript_recent": "转录命中",
        "qa_transcript_matches": "相关时间段",
        "qa_transcript_summary": "总结：",
        "qa_transcript_not_found": "未在转录中找到相关内容。",
        "qa_general_recent": "多轮对话",
        "qa_general_hint": "开启任一功能即可生成回答。",
        "qa_need_selection": "请至少勾选一个回答功能。",
        "qa_question_required": "请输入问题。",
        "language_original": "原始版本",
        "translation_failed": "生成翻译版本时出错：{error}",
        "translation_not_supported": "暂不支持该语言。",
        "qa_hint": "在侧边栏中可随时切换讲座与版本进行问答。",
        "slide_missing": "幻灯片暂不可用",
        "no_translations": "暂无可选版本。",
        "settings_button": "管理与设置",
        "settings_section": "系统设置",
        "settings_llm_model": "LLM 模型名称",
        "settings_llm_base_url": "LLM 接口地址",
        "settings_llm_api_key": "LLM API Key",
        "settings_whisper_model": "Whisper 模型",
        "settings_lecture_root": "讲座存储目录",
        "settings_save": "保存并重启",
        "settings_saved": "设置已保存，应用已重新加载。",
        "settings_close": "关闭",
        "stage_transcription": "录音转文本",
        "stage_outline": "知识梳理",
        "stage_slides": "幻灯片对齐",
        "stage_enrichment": "知识点关联",
    },
    "en": {
        "ui_language": "UI Language",
        "section_manage": "Lecture Management",
        "input_title": "Lecture Title",
        "upload_video": "Upload Lecture Video",
        "upload_pdf": "Upload slides PDF (optional)",
        "translation_select": "Select additional language versions",
        "btn_start": "Start Processing",
        "warn_fill": "Please enter a lecture title and upload a video.",
        "status_processing": "Processing lecture...",
        "status_complete": "Completed",
        "summary_template": "Lecture \"{title}\" processed: {segments} transcript segments · {points} knowledge points · {slides} slide images.",
        "latest_progress": "Latest Processing Details",
        "lecture_list": "Lecture List",
        "no_lectures": "No lectures yet. Upload a video to create one.",
        "create_conflict": "A lecture with the same name already exists. Please choose another name.",
        "delete": "Delete",
        "delete_success": "Lecture \"{title}\" removed.",
        "section_view": "Lecture Explorer",
        "select_lecture": "Select Lecture",
        "select_language": "Select Version",
        "section_video": "Video",
        "no_video": "Video file not found.",
        "section_summary": "Overview",
        "session_topic": "Topic: {topic}",
        "section_points": "Knowledge Points",
        "no_points": "No knowledge points available.",
        "time_range": "Time: {start:.2f}s - {end:.2f}s",
        "summary_label": "Summary:",
        "content_label": "Detailed Notes:",
        "teaching_label": "Teaching Method:",
        "emphasis_label": "Emphasis:",
        "slides_linked": "Related Slides:",
        "section_slides": "Slide Gallery",
        "no_slides": "No slides available.",
        "section_transcript": "Full Transcript",
        "expand_transcript": "Expand transcript",
        "section_qa": "Knowledge Q&A",
        "qa_enable": "Enable Q&A",
        "qa_disabled": "Check the box above to enable Q&A.",
        "qa_no_lectures": "No lecture available.",
        "qa_select_lecture": "Select Lecture",
        "qa_select_language": "Select Version",
        "qa_question": "Question",
        "qa_submit": "Generate Answer",
        "qa_recent": "Recent Answers",
        "qa_feature_transcript": "Use transcript evidence (timestamps + summary)",
        "qa_feature_general": "Use free-form reasoning (multi-turn)",
        "qa_transcript_recent": "Transcript Hits",
        "qa_transcript_matches": "Matching time ranges",
        "qa_transcript_summary": "Summary:",
        "qa_transcript_not_found": "No matching transcript content.",
        "qa_general_recent": "Conversation",
        "qa_general_hint": "Enable at least one feature to see answers.",
        "qa_need_selection": "Please enable at least one answering mode.",
        "qa_question_required": "Please enter a question.",
        "language_original": "Original",
        "translation_failed": "Failed to generate translations: {error}",
        "translation_not_supported": "Language not supported yet.",
        "qa_hint": "Use the sidebar to ask questions while browsing the lecture.",
        "slide_missing": "Slide unavailable",
        "no_translations": "No additional versions available.",
        "settings_button": "Manage & Settings",
        "settings_section": "System Settings",
        "settings_llm_model": "LLM model name",
        "settings_llm_base_url": "LLM base URL",
        "settings_llm_api_key": "LLM API key",
        "settings_whisper_model": "Whisper model",
        "settings_lecture_root": "Lecture storage directory",
        "settings_save": "Save & restart",
        "settings_saved": "Settings saved and app reloaded.",
        "settings_close": "Close",
        "stage_transcription": "Transcription",
        "stage_outline": "Outline & Notes",
        "stage_slides": "Slide Alignment",
        "stage_enrichment": "Enrichment",
    },
}

PIPELINE_STAGES = ["transcription", "outline", "slides", "enrichment"]

UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024
WHISPER_MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]


def persist_temp_upload(uploaded_file) -> tuple[Path, str]:
    temp_dir = LECTURE_ROOT / "_tmp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name or "upload.bin").suffix or ".mp4"
    temp_path = temp_dir / f"{uuid4().hex}{suffix}"
    hasher = hashlib.sha256()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    with temp_path.open("wb") as handle:
        while True:
            chunk = uploaded_file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            handle.write(chunk)
            hasher.update(chunk)
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return temp_path, hasher.hexdigest()


def cleanup_file(path: Path | None):
    if not path:
        return
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def discover_existing_assets(lecture_dir: Path) -> dict:
    assets = {
        "video_path": None,
        "video_hash": None,
        "pdf_path": None,
        "pdf_hash": None,
    }
    if not lecture_dir.exists():
        return assets

    videos_dir = lecture_dir / "videos"
    if videos_dir.exists():
        video_files = [p for p in videos_dir.glob("*") if p.is_file()]
        if video_files:
            latest = max(video_files, key=lambda p: p.stat().st_mtime)
            assets["video_path"] = latest
            try:
                assets["video_hash"] = compute_file_hash(latest)
            except FileNotFoundError:
                pass

    pdf_dir = lecture_dir / "pdf_uploads"
    candidates = []
    if pdf_dir.exists():
        candidates.extend([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    slides_pdf = lecture_dir / "slides.pdf"
    if slides_pdf.exists():
        candidates.append(slides_pdf)
    if candidates:
        latest_pdf = max(candidates, key=lambda p: p.stat().st_mtime)
        assets["pdf_path"] = latest_pdf
        try:
            assets["pdf_hash"] = compute_file_hash(latest_pdf)
        except FileNotFoundError:
            pass
    return assets


def available_translation_codes(record: LectureRecord) -> List[str]:
    codes = set(getattr(record, "translations", []) or [])
    translation_dir = record.path / "translations"
    if translation_dir.exists():
        for child in translation_dir.iterdir():
            if not child.is_dir():
                continue
            outline_file = child / "knowledge_outline_enriched.json"
            segments_file = child / "transcript_segments.json"
            if outline_file.exists() and segments_file.exists():
                codes.add(child.name)
    return sorted(codes)


def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("ui_lang", "zh")
    template = UI_TEXT.get(lang, UI_TEXT["zh"]).get(key, key)
    if kwargs:
        return template.format(**kwargs)
    return template


def load_language_resources(lecture_path_str: str, language: str | None):
    lecture_dir = Path(lecture_path_str)
    lang_key = language or "base"
    assets = build_assets(lecture_dir, language=None if lang_key == "base" else language)
    qa_system = TranscriptQASystem(segments=assets.segments)
    slide_lookup = slides_by_index(assets.slides)
    return assets, qa_system, slide_lookup


@st.cache_resource(show_spinner=True)
def load_resources(lecture_path_str: str, language: str | None):
    return load_language_resources(lecture_path_str, language)


def lectures_signature() -> str:
    LECTURE_ROOT.mkdir(parents=True, exist_ok=True)
    fingerprints: List[str] = []
    for child in sorted(LECTURE_ROOT.iterdir()):
        if not child.is_dir():
            continue
        metadata_path = child / METADATA_FILENAME
        if not metadata_path.exists():
            continue
        stat = metadata_path.stat()
        fingerprints.append(f"{child.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(fingerprints)


@st.cache_data(show_spinner=False)
def cached_lectures(signature: str) -> List[LectureRecord]:
    return list_lectures()


def refresh_caches():
    cached_lectures.clear()
    load_resources.clear()
    st.session_state.pop("last_run_details", None)


def trigger_rerun():
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()


def ensure_selected_lecture(lectures: List[LectureRecord]) -> str | None:
    if not lectures:
        st.session_state.pop("selected_lecture", None)
        return None
    current = st.session_state.get("selected_lecture")
    lecture_ids = [rec.lecture_id for rec in lectures]
    if current not in lecture_ids:
        st.session_state["selected_lecture"] = lecture_ids[0]
    return st.session_state["selected_lecture"]


def delete_lecture(lecture_id: str):
    lecture_dir = get_lecture_dir(lecture_id)
    if lecture_dir.exists():
        shutil.rmtree(lecture_dir, ignore_errors=True)


def render_run_details(details: dict):
    st.subheader(t("latest_progress"))
    st.caption(details.get("summary", ""))
    timings = details.get("timings", {})
    for key, meta in STEP_META.items():
        if key not in timings:
            continue
        label = t(meta["label_key"])
        duration = timings[key]
        header = f"{label} · {duration:.2f}s"
        with st.expander(header, expanded=False):
            metric_key = meta.get("metric_key")
            if metric_key and metric_key in details and meta["metric_label_key"]:
                st.write(f"{t(meta['metric_label_key'])}：{details[metric_key]}")
            else:
                st.write(t("status_complete"))


def render_management_section(lectures: List[LectureRecord]):
    st.subheader(t("section_manage"))
    LECTURE_ROOT.mkdir(parents=True, exist_ok=True)
    app_settings = get_settings()
    title = st.text_input(t("input_title"), placeholder="Lecture title", key="new_lecture_title")
    uploaded = st.file_uploader(t("upload_video"), type=["mp4", "mov", "mkv", "avi"], key="new_lecture_video")
    pdf_uploaded = st.file_uploader(t("upload_pdf"), type=["pdf"], key="new_lecture_pdf")
    translation_options = list(LANGUAGE_CONFIG.keys())
    translation_choices = st.multiselect(
        t("translation_select"),
        options=translation_options,
        format_func=lambda code: LANGUAGE_CONFIG[code]["label"],
        key="new_translation_languages",
    )

    run_disabled = uploaded is None or not title.strip()
    if st.button(t("btn_start"), use_container_width=True, disabled=run_disabled):
        if uploaded is None or not title.strip():
            st.warning(t("warn_fill"))
        else:
            lecture_title = title.strip()
            lecture_id = slugify(lecture_title)
            lecture_dir = LECTURE_ROOT / lecture_id
            temp_path: Path | None = None
            pdf_temp_path: Path | None = None
            pdf_hash: str | None = None
            try:
                temp_path, _ = persist_temp_upload(uploaded)
                if pdf_uploaded is not None:
                    pdf_temp_path, _ = persist_temp_upload(pdf_uploaded)

                existing_metadata = load_metadata(lecture_dir) if lecture_dir.exists() else None
                existing_assets = {
                    "video_path": existing_metadata.video_path if existing_metadata and existing_metadata.video_path else None,
                    "pdf_path": existing_metadata.pdf_path if existing_metadata and existing_metadata.pdf_path else None,
                }
                if lecture_dir.exists() and existing_metadata is None:
                    discovered = discover_existing_assets(lecture_dir)
                    for key, value in discovered.items():
                        if existing_assets.get(key) is None and value is not None:
                            existing_assets[key] = value

                lecture_dir.mkdir(parents=True, exist_ok=True)

                video_dir = lecture_dir / "videos"
                video_dir.mkdir(exist_ok=True)
                original_name = uploaded.name or "lecture.mp4"
                target_path = video_dir / f"{Path(original_name).stem}_{uuid4().hex[:6]}{Path(original_name).suffix}"
                shutil.move(str(temp_path), target_path)
                temp_path = None
                video_path = target_path

                pdf_storage_path = None
                if pdf_temp_path is not None:
                    pdf_dir = lecture_dir / "pdf_uploads"
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    pdf_name = Path(pdf_uploaded.name or "slides.pdf").stem
                    target_pdf = pdf_dir / f"{slugify(pdf_name)}_{uuid4().hex[:6]}.pdf"
                    shutil.move(str(pdf_temp_path), target_pdf)
                    pdf_storage_path = target_pdf
                    pdf_temp_path = None
                elif existing_assets.get("pdf_path") and existing_assets["pdf_path"].exists():
                    pdf_storage_path = existing_assets["pdf_path"]

                stage_labels = {stage: t(f"stage_{stage}") for stage in PIPELINE_STAGES}
                with st.status(t("status_processing"), expanded=True) as status:
                    stage_container = st.container()
                    stage_placeholders = {stage: stage_container.empty() for stage in PIPELINE_STAGES}

                    def progress(message: str | dict):
                        if isinstance(message, dict):
                            stage = message.get("stage")
                            action = message.get("action")
                            detail = message.get("detail", "")
                            placeholder = stage_placeholders.get(stage)
                            label = stage_labels.get(stage, stage)
                            if placeholder is not None:
                                if action == "start":
                                    placeholder.info(f"{label} · {detail}")
                                elif action == "complete":
                                    placeholder.success(f"{label} · {detail}")
                                elif action == "skip":
                                    placeholder.warning(f"{label} · {detail}")
                                else:
                                    placeholder.write(f"{label} · {detail}")
                                return
                        status.write(message)

                    result = run_pipeline(
                        video_path=video_path,
                        pdf_path=pdf_storage_path,
                        output_dir=lecture_dir,
                        lecture_id=lecture_id,
                        whisper_model=app_settings.get("whisper_model") or app_config.DEFAULT_WHISPER_MODEL,
                        progress_callback=progress,
                    )
                    metadata_pdf_path = Path(result.slides_pdf_path) if result.slides_pdf_path else pdf_storage_path
                    save_metadata(
                        lecture_dir,
                        lecture_id=lecture_id,
                        title=lecture_title,
                        video_path=video_path,
                        source_language=result.language or "unknown",
                        translations=[],
                        video_hash=None,
                        pdf_path=metadata_pdf_path,
                        pdf_hash=None,
                    )
                    status.update(label=t("status_complete"), state="complete")

                summary = t(
                    "summary_template",
                    title=lecture_title,
                    segments=result.segment_count,
                    points=len(result.outline.knowledge_points),
                    slides=result.slide_count,
                )
                st.success(summary)
                st.session_state["selected_lecture"] = lecture_id
                st.session_state["last_run_details"] = {
                    "lecture_id": lecture_id,
                    "summary": summary,
                    "timings": result.timings,
                    "segments": result.segment_count,
                    "knowledge_points": len(result.outline.knowledge_points),
                    "slides": result.slide_count,
                }

                additional_langs = [lang for lang in translation_choices if lang != result.language]
                if additional_langs:
                    try:
                        translate_outputs(lecture_dir, additional_langs)
                        metadata = load_metadata(lecture_dir)
                        existing = metadata.translations if metadata else []
                        merged = sorted(set(existing + additional_langs))
                        update_translations(lecture_dir, merged)
                    except Exception as exc:  # pragma: no cover
                        st.error(t("translation_failed", error=str(exc)))

                refresh_caches()
                trigger_rerun()
            finally:
                cleanup_file(temp_path)
                cleanup_file(pdf_temp_path)

    if "last_run_details" in st.session_state:
        render_run_details(st.session_state["last_run_details"])


def render_lecture_list(lectures: List[LectureRecord]):
    st.subheader(t("lecture_list"))
    if not lectures:
        st.info(t("no_lectures"))
        return

    for record in lectures:
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(
                f"**{record.title}** ({record.lecture_id})  \n"
                f"{record.created_at.strftime('%Y-%m-%d %H:%M:%S')} → {record.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            source_code = getattr(record, "source_language", "unknown")
            lang_label = LANGUAGE_CONFIG.get(source_code, {}).get("label", t("language_original"))
            record_translations = available_translation_codes(record)
            translations = ", ".join(LANGUAGE_CONFIG.get(code, {}).get("label", code) for code in record_translations)
            st.caption(f"{t('language_original')}: {lang_label} · {translations or t('no_translations')}")
        with cols[1]:
            if st.button(t("delete"), key=f"delete_{record.lecture_id}", use_container_width=True):
                delete_lecture(record.lecture_id)
                if st.session_state.get("selected_lecture") == record.lecture_id:
                    st.session_state.pop("selected_lecture", None)
                refresh_caches()
                st.success(t("delete_success", title=record.title))
                trigger_rerun()


def render_settings_panel():
    st.subheader(t("settings_section"))
    settings = get_settings()
    llm_model = settings.get("llm_model", "")
    llm_base_url = settings.get("llm_base_url", "")
    llm_api_key = settings.get("llm_api_key", "")
    whisper_model = settings.get("whisper_model", app_config.DEFAULT_WHISPER_MODEL)
    lecture_root = settings.get("lecture_root", "data/lectures")
    whisper_index = WHISPER_MODEL_CHOICES.index(whisper_model) if whisper_model in WHISPER_MODEL_CHOICES else 0

    with st.form("settings_form", clear_on_submit=False):
        llm_model_value = st.text_input(t("settings_llm_model"), value=llm_model)
        llm_base_url_value = st.text_input(t("settings_llm_base_url"), value=llm_base_url)
        llm_api_key_value = st.text_input(t("settings_llm_api_key"), value=llm_api_key, type="password")
        whisper_model_value = st.selectbox(
            t("settings_whisper_model"),
            options=WHISPER_MODEL_CHOICES,
            index=whisper_index,
        )
        lecture_root_value = st.text_input(t("settings_lecture_root"), value=lecture_root)
        submitted = st.form_submit_button(t("settings_save"))

    if submitted:
        update_settings(
            {
                "llm_model": llm_model_value.strip(),
                "llm_base_url": llm_base_url_value.strip(),
                "llm_api_key": llm_api_key_value.strip(),
                "whisper_model": whisper_model_value,
                "lecture_root": lecture_root_value.strip() or "data/lectures",
            }
        )
        app_config.refresh_runtime_settings()
        refresh_caches()
        st.session_state["settings_flash"] = t("settings_saved")
        trigger_rerun()


def render_control_panel(lectures: List[LectureRecord]):
    if st.button(f"⚙️ {t('settings_button')}", key="open_control_panel", use_container_width=True):
        st.session_state["control_panel_open"] = True

    if st.session_state.get("control_panel_open"):
        with st.container(border=True):
            st.markdown(f"### {t('settings_button')}")
            render_management_section(lectures)
            st.divider()
            render_lecture_list(lectures)
            st.divider()
            render_settings_panel()
            if st.button(t("settings_close"), key="close_control_panel", use_container_width=True):
                st.session_state["control_panel_open"] = False
                trigger_rerun()


def get_language_options(record: LectureRecord):
    options = []
    base_label = LANGUAGE_CONFIG.get(getattr(record, "source_language", "unknown"), {}).get(
        "label", t("language_original")
    )
    options.append(("base", f"{base_label} ({t('language_original')})"))
    for lang in available_translation_codes(record):
        label = LANGUAGE_CONFIG.get(lang, {}).get("label", lang)
        options.append((lang, label))
    return options


def render_view_section(lectures: List[LectureRecord]):
    st.header(t("section_view"))
    if not lectures:
        st.info(t("no_lectures"))
        return

    selected_id = ensure_selected_lecture(lectures)
    if selected_id is None:
        st.info(t("no_lectures"))
        return

    lecture_map = {rec.lecture_id: rec for rec in lectures}
    labels = {rec.lecture_id: f"{rec.title} ({rec.lecture_id})" for rec in lectures}
    selected_lecture = st.selectbox(
        t("select_lecture"),
        options=list(labels.keys()),
        format_func=lambda key: labels[key],
        index=list(labels.keys()).index(selected_id),
        key="viewer_select",
    )
    st.session_state["selected_lecture"] = selected_lecture
    active_lecture = lecture_map[selected_lecture]

    lang_options = get_language_options(active_lecture)
    lang_keys = [opt[0] for opt in lang_options]
    lang_labels = {opt[0]: opt[1] for opt in lang_options}
    selected_lang = st.selectbox(
        t("select_language"),
        options=lang_keys,
        format_func=lambda key: lang_labels[key],
        key="view_language_select",
    )
    language_param = None if selected_lang == "base" else selected_lang

    try:
        assets, _, slide_lookup = load_resources(
            str(active_lecture.path.resolve()),
            language_param,
        )
    except FileNotFoundError:
        st.warning(t("no_lectures"))
        return

    st.subheader(t("section_video"))
    if active_lecture.video_path and active_lecture.video_path.exists():
        st.video(str(active_lecture.video_path))
    else:
        st.info(t("no_video"))

    st.subheader(t("section_summary"))
    st.markdown(t("session_topic", topic=assets.outline.session_topic or "-"))
    st.write(assets.outline.overall_summary or "-")

    st.subheader(t("section_points"))
    if not assets.outline.knowledge_points:
        st.info(t("no_points"))
    else:
        options = [f"{idx + 1}. {kp.title or '—'}" for idx, kp in enumerate(assets.outline.knowledge_points)]
        kp_index = st.selectbox(
            t("section_points"),
            range(len(options)),
            format_func=lambda i: options[i],
            key="knowledge_point_select",
        )
        kp = assets.outline.knowledge_points[kp_index]
        st.markdown(t("time_range", start=kp.start_time, end=kp.end_time))
        st.markdown(f"**{t('summary_label')}** {kp.summary or '—'}")
        st.markdown(f"**{t('content_label')}**\n{kp.content or '—'}")
        st.markdown(f"**{t('teaching_label')}** {kp.teaching_method or '—'}")
        st.markdown(f"**{t('emphasis_label')}** {kp.emphasis or '—'}")
        if kp.slides:
            st.markdown(f"**{t('slides_linked')}**")
            cols = st.columns(min(3, len(kp.slides)))
            for idx, slide_id in enumerate(kp.slides):
                slide = slide_lookup.get(slide_id)
                if slide and Path(slide.image_path).exists():
                    with cols[idx % len(cols)]:
                        st.image(slide.image_path, caption=f"Slide {slide_id}")
                else:
                    st.caption(t("slide_missing"))

    st.subheader(t("section_slides"))
    with st.expander(t("section_slides"), expanded=False):
        if not assets.slides:
            st.info(t("no_slides"))
        else:
            cols = st.columns(3)
            for idx, slide in enumerate(assets.slides):
                with cols[idx % 3]:
                    if Path(slide.image_path).exists():
                        st.image(slide.image_path, caption=f"Slide {slide.slide_index} ({slide.start_time:.1f}s)")

    st.subheader(t("section_transcript"))
    with st.expander(t("expand_transcript"), expanded=False):
        for segment in assets.segments:
            st.markdown(f"- **[{segment.start:.2f}s - {segment.end:.2f}s]** {segment.text}")


def render_qa_sidebar(lectures: List[LectureRecord]):
    with st.sidebar:
        st.header(t("section_qa"))
        if not lectures:
            st.info(t("qa_no_lectures"))
            return

        lecture_map = {rec.lecture_id: rec for rec in lectures}
        labels = {rec.lecture_id: f"{rec.title} ({rec.lecture_id})" for rec in lectures}
        selected_lecture = st.selectbox(
            t("qa_select_lecture"),
            options=list(labels.keys()),
            format_func=lambda key: labels[key],
            key="qa_lecture_select",
        )
        record = lecture_map[selected_lecture]
        lang_options = get_language_options(record)
        lang_keys = [opt[0] for opt in lang_options]
        lang_labels = {opt[0]: opt[1] for opt in lang_options}
        selected_lang = st.selectbox(
            t("qa_select_language"),
            options=lang_keys,
            format_func=lambda key: lang_labels[key],
            key="qa_language_select",
        )
        language_param = None if selected_lang == "base" else selected_lang

        try:
            _, qa_system, _ = load_resources(
                str(record.path.resolve()),
                language_param,
            )
        except FileNotFoundError:
            st.warning(t("qa_no_lectures"))
            return

        question = st.text_area(t("qa_question"), height=100, key="qa_question")
        use_transcript = st.checkbox(t("qa_feature_transcript"), value=True, key="qa_use_transcript")
        use_general = st.checkbox(t("qa_feature_general"), value=True, key="qa_use_general")

        history_key = (selected_lecture, selected_lang)
        transcript_histories = st.session_state.setdefault("transcript_history", {})
        transcript_history = transcript_histories.setdefault(history_key, [])
        general_histories = st.session_state.setdefault("general_chat_history", {})
        general_history = general_histories.setdefault(history_key, [])
        general_agent = GeneralChatAgent()

        if st.button(t("qa_submit"), use_container_width=True, key="qa_submit"):
            if not question.strip():
                st.warning(t("qa_question_required"))
            elif not (use_transcript or use_general):
                st.warning(t("qa_need_selection"))
            else:
                if use_transcript:
                    result = qa_system.answer(question)
                    transcript_history.insert(
                        0,
                        {
                            "question": question,
                            "status": result.status,
                            "answer": result.answer,
                            "matches": [
                                {
                                    "start_time": match.start_time,
                                    "end_time": match.end_time,
                                    "excerpt": match.excerpt,
                                    "summary": match.summary,
                                }
                                for match in result.matches
                            ],
                        },
                    )
                    transcript_histories[history_key] = transcript_history[:5]

                if use_general:
                    response = general_agent.answer(question, general_history)
                    general_history.append({"role": "user", "content": question})
                    general_history.append({"role": "assistant", "content": response.answer})
                    if len(general_history) > 20:
                        general_history[:] = general_history[-20:]
                    general_histories[history_key] = general_history

                st.session_state["transcript_history"] = transcript_histories
                st.session_state["general_chat_history"] = general_histories
                trigger_rerun()

        if transcript_history:
            st.subheader(t("qa_transcript_recent"))
            for entry in transcript_history[:5]:
                st.markdown(f"**Q:** {entry['question']}")
                matches = entry.get("matches", [])
                if not matches:
                    st.caption(t("qa_transcript_not_found"))
                    st.write(entry.get("answer", ""))
                    continue
                st.markdown(f"**{t('qa_transcript_matches')}**")
                for match in matches:
                    st.caption(
                        f"[{match['start_time']:.2f}s - {match['end_time']:.2f}s] {match['excerpt']}"
                    )
                    st.write(match["summary"])
                st.markdown(f"**{t('qa_transcript_summary')}** {entry.get('answer', '')}")

        if general_history:
            st.subheader(t("qa_general_recent"))
            for msg in general_history[-6:]:
                prefix = "Q" if msg.get("role") == "user" else "A"
                st.markdown(f"**{prefix}:** {msg.get('content', '')}")

        if not transcript_history and not general_history:
            st.info(t("qa_general_hint"))


def main():
    st.set_page_config(
        page_title=UIConfig.PAGE_TITLE,
        page_icon=UIConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    default_lang = st.session_state.get("ui_lang", "zh")
    st.session_state["ui_lang"] = st.sidebar.selectbox(
        t("ui_language"),
        options=["zh", "en"],
        format_func=lambda code: "中文" if code == "zh" else "English",
        index=0 if default_lang == "zh" else 1,
    )

    flash_message = st.session_state.pop("settings_flash", None)
    if flash_message:
        st.success(flash_message)

    lectures = cached_lectures(lectures_signature())
    render_control_panel(lectures)
    st.divider()
    render_view_section(lectures)
    render_qa_sidebar(lectures)


if __name__ == "__main__":
    main()
