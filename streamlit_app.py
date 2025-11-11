from __future__ import annotations

import shutil
from pathlib import Path
from typing import List
from uuid import uuid4

import streamlit as st

from app import build_assets
from app.config import LECTURE_ROOT, UIConfig
from app.knowledge_base import slides_by_index
from app.lectures import (
    LectureRecord,
    get_lecture_dir,
    list_lectures,
    load_metadata,
    save_metadata,
    slugify,
    update_translations,
)
from app.pipeline.runner import run_pipeline
from app.qa import KnowledgeQASystem
from app.translation import LANGUAGE_CONFIG, translate_outputs
from app.vector_store import build_faiss_store

STEP_META = {
    "transcription": {"label_key": "step_transcription", "metric_key": "segments", "metric_label_key": "metric_segments"},
    "outline": {"label_key": "step_outline", "metric_key": "knowledge_points", "metric_label_key": "metric_points"},
    "slides": {"label_key": "step_slides", "metric_key": "slides", "metric_label_key": "metric_slides"},
    "enrichment": {"label_key": "step_enrichment", "metric_key": None, "metric_label_key": ""},
}

AVAILABLE_TRANSLATIONS = list(LANGUAGE_CONFIG.keys())

UI_TEXT = {
    "zh": {
        "ui_language": "界面语言",
        "section_manage": "讲座管理",
        "input_title": "讲座名称",
        "upload_video": "上传课堂视频",
        "translation_select": "选择需要同步生成的其他语言版本",
        "btn_start": "开始处理",
        "warn_fill": "请填写讲座名称并上传视频。",
        "status_processing": "正在处理讲座...",
        "status_complete": "处理完成",
        "summary_template": "讲座《{title}》处理完成：{segments} 条转录片段 · {points} 个知识点 · {slides} 张幻灯片。",
        "latest_progress": "最新处理进度",
        "lecture_list": "讲座列表",
        "no_lectures": "当前还没有任何讲座，上传视频即可自动创建。",
        "delete": "删除",
        "delete_success": "已删除讲座《{title}》",
        "section_view": "课堂知识浏览",
        "select_lecture": "选择讲座",
        "select_language": "选择版本",
        "faiss_status": "FAISS 索引：{status}",
        "section_video": "课堂视频",
        "no_video": "未找到该讲座对应的视频文件。",
        "section_summary": "整体梳理",
        "session_topic": "主题：{topic}",
        "section_points": "知识点",
        "no_points": "暂无知识点数据。",
        "time_range": "时间段：{start:.2f}s - {end:.2f}s",
        "summary_label": "摘要：",
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
        "qa_strategy": "检索策略",
        "qa_mode_bm25": "BM25 词法检索",
        "qa_mode_faiss": "FAISS 向量检索",
        "qa_use_llm": "使用 LLM 生成回答",
        "qa_submit": "生成回答",
        "qa_recent": "最近的回答",
        "qa_no_hits": "未检索到相关片段。",
        "language_original": "原始版本",
        "translation_failed": "生成翻译版本时出错：{error}",
        "translation_not_supported": "暂不支持该语言。",
        "qa_enable_label": "启用问答助手",
        "qa_history_label": "关联片段",
        "qa_empty": "暂无回答记录。",
        "qa_hint": "在侧边栏中可随时切换讲座与版本进行问答。",
        "slide_missing": "幻灯片暂不可用",
        "no_translations": "暂无可选版本。",
    },
    "en": {
        "ui_language": "UI Language",
        "section_manage": "Lecture Management",
        "input_title": "Lecture Title",
        "upload_video": "Upload Lecture Video",
        "translation_select": "Select additional language versions",
        "btn_start": "Start Processing",
        "warn_fill": "Please enter a lecture title and upload a video.",
        "status_processing": "Processing lecture...",
        "status_complete": "Completed",
        "summary_template": "Lecture \"{title}\" processed: {segments} transcript segments · {points} knowledge points · {slides} slide images.",
        "latest_progress": "Latest Processing Details",
        "lecture_list": "Lecture List",
        "no_lectures": "No lectures yet. Upload a video to create one.",
        "delete": "Delete",
        "delete_success": "Lecture \"{title}\" removed.",
        "section_view": "Lecture Explorer",
        "select_lecture": "Select Lecture",
        "select_language": "Select Version",
        "faiss_status": "FAISS Index: {status}",
        "section_video": "Video",
        "no_video": "Video file not found.",
        "section_summary": "Overview",
        "session_topic": "Topic: {topic}",
        "section_points": "Knowledge Points",
        "no_points": "No knowledge points available.",
        "time_range": "Time: {start:.2f}s - {end:.2f}s",
        "summary_label": "Summary:",
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
        "qa_strategy": "Retrieval Strategy",
        "qa_mode_bm25": "BM25 lexical",
        "qa_mode_faiss": "FAISS vector",
        "qa_use_llm": "Use LLM to answer",
        "qa_submit": "Generate Answer",
        "qa_recent": "Recent Answers",
        "qa_no_hits": "No related segments found.",
        "language_original": "Original",
        "translation_failed": "Failed to generate translations: {error}",
        "translation_not_supported": "Language not supported yet.",
        "qa_enable_label": "Enable Q&A Assistant",
        "qa_history_label": "Retrieved Segments",
        "qa_empty": "No answers yet.",
        "qa_hint": "Use the sidebar to ask questions while browsing the lecture.",
        "slide_missing": "Slide unavailable",
        "no_translations": "No additional versions available.",
    },
}


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
    faiss_index_dir = (
        lecture_dir / "faiss_index"
        if lang_key == "base"
        else lecture_dir / "translations" / language / "faiss_index"
    )
    faiss_retriever, faiss_status = build_faiss_store(assets.documents, persist_path=faiss_index_dir)
    qa_system = KnowledgeQASystem(
        documents=assets.documents,
        faiss_retriever=faiss_retriever,
    )
    slide_lookup = slides_by_index(assets.slides)
    return assets, qa_system, faiss_status, slide_lookup


@st.cache_resource(show_spinner=True)
def load_resources(lecture_path_str: str, language: str | None):
    return load_language_resources(lecture_path_str, language)


@st.cache_data(show_spinner=False)
def cached_lectures() -> List[LectureRecord]:
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
    st.header(t("section_manage"))
    LECTURE_ROOT.mkdir(parents=True, exist_ok=True)

    title = st.text_input(t("input_title"), placeholder="Lecture title", key="new_lecture_title")
    uploaded = st.file_uploader(t("upload_video"), type=["mp4", "mov", "mkv", "avi"], key="new_lecture_video")
    translation_choices = st.multiselect(
        t("translation_select"),
        options=AVAILABLE_TRANSLATIONS,
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
            if lecture_dir.exists():
                lecture_id = f"{lecture_id}-{uuid4().hex[:4]}"
                lecture_dir = LECTURE_ROOT / lecture_id
            lecture_dir.mkdir(parents=True, exist_ok=True)
            video_dir = lecture_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            original_name = uploaded.name or "lecture.mp4"
            target_path = video_dir / f"{Path(original_name).stem}_{uuid4().hex[:6]}{Path(original_name).suffix}"
            with target_path.open("wb") as f:
                f.write(uploaded.getbuffer())

            with st.status(t("status_processing"), expanded=True) as status:
                def progress(message: str):
                    status.write(message)

                result = run_pipeline(
                    video_path=target_path,
                    output_dir=lecture_dir,
                    lecture_id=lecture_id,
                    progress_callback=progress,
                )
                save_metadata(
                    lecture_dir,
                    lecture_id=lecture_id,
                    title=lecture_title,
                    video_path=target_path,
                    source_language=result.language or "unknown",
                    translations=[],
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

    if "last_run_details" in st.session_state:
        render_run_details(st.session_state["last_run_details"])

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
            record_translations = getattr(record, "translations", []) or []
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


def get_language_options(record: LectureRecord):
    options = []
    base_label = LANGUAGE_CONFIG.get(getattr(record, "source_language", "unknown"), {}).get(
        "label", t("language_original")
    )
    options.append(("base", f"{base_label} ({t('language_original')})"))
    for lang in getattr(record, "translations", []) or []:
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
        assets, _, faiss_status, slide_lookup = load_resources(
            str(active_lecture.path.resolve()),
            language_param,
        )
    except FileNotFoundError:
        st.warning(t("no_lectures"))
        return

    faiss_label = "✅ OK" if faiss_status.ready else f"⚠️ {faiss_status.message}"
    st.caption(t("faiss_status", status=faiss_label))

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
        enabled = st.checkbox(t("qa_enable"), value=True, key="qa_enabled")
        if not enabled:
            st.info(t("qa_disabled"))
            return
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
        if not lang_options:
            st.info(t("no_translations"))
            return
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
            assets, qa_system, faiss_status, _ = load_resources(
                str(record.path.resolve()),
                language_param,
            )
        except FileNotFoundError:
            st.warning(t("qa_no_lectures"))
            return

        faiss_label = "✅ OK" if faiss_status.ready else f"⚠️ {faiss_status.message}"
        st.caption(t("faiss_status", status=faiss_label))

        question = st.text_area(t("qa_question"), height=100, key="qa_question")
        retrieval_mode = st.radio(
            t("qa_strategy"),
            ["bm25", "faiss"],
            format_func=lambda mode: t("qa_mode_faiss") if mode == "faiss" else t("qa_mode_bm25"),
            key="qa_mode",
        )
        use_llm = st.checkbox(t("qa_use_llm"), value=qa_system.llm is not None, key="qa_use_llm")

        history_key = (selected_lecture, selected_lang)
        qa_history = st.session_state.setdefault("qa_history", {})
        lecture_history = qa_history.setdefault(history_key, [])

        if st.button(t("qa_submit"), use_container_width=True, key="qa_submit"):
            if question.strip():
                result = qa_system.answer(question, mode=retrieval_mode, use_llm=use_llm)
                lecture_history.insert(
                    0,
                    {
                        "question": question,
                        "answer": result.answer,
                        "hits": result.hits,
                    },
                )
                qa_history[history_key] = lecture_history[:5]
                st.session_state["qa_history"] = qa_history
                trigger_rerun()

        if lecture_history:
            st.subheader(t("qa_recent"))
            for entry in lecture_history[:5]:
                st.markdown(f"**Q:** {entry['question']}")
                st.write(entry["answer"])
                with st.expander(t("qa_history_label"), expanded=False):
                    if not entry["hits"]:
                        st.write(t("qa_no_hits"))
                    for hit in entry["hits"]:
                        meta = hit.document.metadata or {}
                        st.caption(
                            f"[{meta.get('start_time', 0):.2f}-{meta.get('end_time', 0):.2f}s] {hit.document.page_content}"
                        )
        else:
            st.info(t("qa_hint"))


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

    lectures = cached_lectures()
    render_management_section(lectures)
    st.divider()
    render_view_section(lectures)
    render_qa_sidebar(lectures)


if __name__ == "__main__":
    main()
