"""
Hybrid Pipeline News Demo
Lathifah Sahda · NRP 5025221159 · Teknik Informatika ITS
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pipeline.preprocessor import run_preprocessing
from pipeline.llm_converter import run_all_models
from pipeline.postprocessor import run_postprocessing
from pipeline.evaluator import run_evaluation

st.set_page_config(
    page_title="Hybrid Pipeline — Narrative to Procedural",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Reset streamlit defaults */
.block-container { padding-top: 2rem; max-width: 1200px; }
#MainMenu, footer, header { visibility: hidden; }

/* Page header */
.page-header {
    border-bottom: 1px solid #e2e2e2;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-size: 1.4rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: #111;
    margin: 0 0 0.3rem 0;
}
.page-header p {
    font-size: 0.82rem;
    color: #888;
    margin: 0;
    font-weight: 300;
}

/* Section label */
.section-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.75rem;
}

/* Pipeline steps bar */
.pipeline-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2rem;
    border: 1px solid #e2e2e2;
    border-radius: 6px;
    overflow: hidden;
}
.pipeline-node {
    flex: 1;
    padding: 0.6rem 0.8rem;
    font-size: 0.75rem;
    color: #666;
    background: #fafafa;
    text-align: center;
    border-right: 1px solid #e2e2e2;
    font-family: 'IBM Plex Mono', monospace;
}
.pipeline-node:last-child { border-right: none; }
.pipeline-node.active {
    background: #111;
    color: #fff;
}

/* Model columns */
.model-header {
    font-size: 0.8rem;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.5rem 0;
    border-bottom: 2px solid;
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
}
.model-header.groq  { border-color: #d97706; color: #d97706; }
.model-header.gpt   { border-color: #16a34a; color: #16a34a; }
.model-header.claude { border-color: #2563eb; color: #2563eb; }

/* Metric chips */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.5rem 0 0.75rem;
}
.chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    border: 1px solid;
}
.chip-green { background: #f0fdf4; border-color: #86efac; color: #15803d; }
.chip-amber { background: #fffbeb; border-color: #fcd34d; color: #b45309; }
.chip-red   { background: #fef2f2; border-color: #fca5a5; color: #b91c1c; }
.chip-gray  { background: #f9fafb; border-color: #d1d5db; color: #6b7280; }

/* Output box */
.output-box {
    background: #f8f8f8;
    border: 1px solid #e8e8e8;
    border-radius: 6px;
    padding: 1rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.9;
    color: #222;
    white-space: pre-wrap;
    max-height: 320px;
    overflow-y: auto;
}

/* Winner card */
.winner-card {
    border: 1px solid #e2e2e2;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    background: #fff;
    margin-bottom: 0.5rem;
}
.winner-card .metric-name {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #999;
    font-weight: 500;
    margin-bottom: 0.2rem;
}
.winner-card .winner-name {
    font-size: 0.85rem;
    font-weight: 500;
    color: #111;
    font-family: 'IBM Plex Mono', monospace;
}

/* Error box */
.error-box {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.78rem;
    color: #991b1b;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.5;
}

/* Table styling */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
}
.comparison-table th {
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid #111;
    font-weight: 500;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #444;
}
.comparison-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #f0f0f0;
    color: #333;
}
.comparison-table tr:last-child td { border-bottom: none; }

/* Insight row */
.insight-row {
    border-left: 3px solid #111;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    background: #fafafa;
    font-size: 0.83rem;
    color: #333;
    line-height: 1.6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #e8e8e8;
}
.sidebar-title {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #aaa;
    margin: 1rem 0 0.5rem;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    padding: 0.3rem 0;
    color: #444;
}
.dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.dot-on  { background: #16a34a; }
.dot-off { background: #d1d5db; }

/* Entity tag */
.ent-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
    margin: 0.1rem;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────────────────────

def chip(label, value, suffix=""):
    try:
        v = float(value)
        if v >= 70:   cls = "chip-green"
        elif v >= 40: cls = "chip-amber"
        else:         cls = "chip-red"
        display = f"{v:.1f}{suffix}"
    except Exception:
        cls = "chip-gray"
        display = str(value)
    return f'<span class="chip {cls}">{label} {display}</span>'

def model_label(key):
    return {
        "groq":   "Groq / Llama 3.3 70B",
        "gpt":    "OpenAI / gpt-4o-mini",
        "claude": "Anthropic / Claude Haiku"
    }.get(key, key)

def get_winners(post, evl):
    models = ["groq", "gpt", "claude"]
    w = {}
    ep    = {m: post[m].get("entity_pass_rate", 0) for m in models if m in post}
    rouge = {m: evl[m].get("rouge", {}).get("rougeL_f", 0)*100 for m in models if m in evl}
    bert  = {m: evl[m].get("bertscore", {}).get("f1", 0)*100 for m in models if m in evl}
    qual  = {m: evl[m].get("overall_quality", 0) for m in models if m in evl}
    if ep:    w["Entity Preservation"] = max(ep, key=ep.get)
    if rouge: w["ROUGE-L"]             = max(rouge, key=rouge.get)
    if bert:  w["BERTScore F1"]        = max(bert, key=bert.get)
    if qual:  w["Overall Quality"]     = max(qual, key=qual.get)
    return w

SAMPLE = """Apple Inc. announced on Tuesday the launch of its new iPhone 16 series,
featuring advanced AI capabilities powered by Apple Intelligence. CEO Tim Cook unveiled the
devices at Apple Park in Cupertino, California, describing the release as a significant
milestone for the company.

The new lineup includes the iPhone 16, iPhone 16 Plus, iPhone 16 Pro, and iPhone 16 Pro Max.
Cook stated that the devices incorporate the A18 chip, which delivers 40 percent faster
performance compared to the previous generation. Apple partnered with OpenAI to integrate
ChatGPT directly into the operating system.

Pre-orders began on September 13, 2024, with devices shipping to customers the following
week. The base model is priced at $799, while the Pro Max starts at $1,199. Apple reported
that pre-order demand exceeded all previous records within the first 24 hours.

The company also announced iOS 18.1, which will be required to activate Apple Intelligence
features. Cook confirmed that the AI features will roll out gradually through software updates
over the coming months. Analysts from Morgan Stanley estimated that Apple could sell up to
90 million units in the first quarter following the launch."""

# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-title">API Keys</p>', unsafe_allow_html=True)

    groq_key = st.text_input(
        "Groq", value=get_secret("GROQ_API_KEY"),
        type="password", placeholder="gsk_..."
    )
    openai_key = st.text_input(
        "OpenAI", value=get_secret("OPENAI_API_KEY"),
        type="password", placeholder="sk-..."
    )
    anthropic_key = st.text_input(
        "Anthropic", value=get_secret("ANTHROPIC_API_KEY"),
        type="password", placeholder="sk-ant-..."
    )

    st.markdown('<p class="sidebar-title">Status</p>', unsafe_allow_html=True)
    active_models = []
    for key, label, mk in [
        (groq_key, "Groq", "groq"),
        (openai_key, "OpenAI", "gpt"),
        (anthropic_key, "Anthropic", "claude")
    ]:
        dot = "dot-on" if key else "dot-off"
        st.markdown(
            f'<div class="status-row"><span class="dot {dot}"></span>{label}</div>',
            unsafe_allow_html=True
        )
        if key:
            active_models.append(mk)

    st.markdown('<p class="sidebar-title">Research</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.75rem;color:#aaa;line-height:1.7;">'
        'Hybrid Pipeline for Narrative-to-Procedural Text Conversion<br>'
        'Lathifah Sahda · 5025221159<br>'
        'Teknik Informatika ITS · 2026</p>',
        unsafe_allow_html=True
    )

# ── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
    <h1>Hybrid Pipeline — Narrative to Procedural</h1>
    <p>Comparative evaluation · Groq · OpenAI · Anthropic · Constraint-based LLM approach</p>
</div>
""", unsafe_allow_html=True)

# Pipeline bar
st.markdown("""
<div class="pipeline-bar">
    <div class="pipeline-node">01 · Input</div>
    <div class="pipeline-node">02 · Pre-processing</div>
    <div class="pipeline-node">03 · LLM Conversion</div>
    <div class="pipeline-node">04 · Post-processing</div>
    <div class="pipeline-node">05 · Evaluation</div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-label">Input Article</p>', unsafe_allow_html=True)

col_text, col_meta = st.columns([3, 1])

with col_text:
    article_text = st.text_area(
        label="article",
        value=SAMPLE,
        height=220,
        label_visibility="collapsed",
        placeholder="Paste a news article in English (200–1000 words)..."
    )

with col_meta:
    if article_text:
        wc = len(article_text.split())
        if 200 <= wc <= 1000:
            st.success(f"{wc} words — valid range")
        elif wc < 200:
            st.warning(f"{wc} words — too short (min 200)")
        else:
            st.warning(f"{wc} words — too long (max 1000)")
    st.caption("Recommended domains: business, technology, practical guides.")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    run_btn = st.button(
        "Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=len(active_models) == 0
    )

if not active_models:
    st.info("Add at least one API key in the sidebar to run the pipeline.")

# ── PIPELINE ──────────────────────────────────────────────────────────────────

if run_btn and article_text and active_models:

    # ── FASE 2 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">02 · Pre-processing</p>', unsafe_allow_html=True)

    with st.spinner("Running NER, action verb extraction, temporal analysis..."):
        t0 = time.time()
        pre = run_preprocessing(article_text)
        pre_time = time.time() - t0

    st.caption(f"Completed in {pre_time:.2f}s — {pre['word_count']} words, {pre['sentence_count']} sentences, {len(pre['action_sequence'])} action verbs detected")

    with st.expander("View pre-processing detail", expanded=False):
        t1, t2, t3, t4 = st.tabs(["Summary", "Entities", "Action Verbs", "Temporal"])

        with t1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Words", pre["word_count"])
            c2.metric("Sentences", pre["sentence_count"])
            c3.metric("Action Verbs", len(pre["action_sequence"]))
            ents = pre["constraints"]["mandatory_entities"]
            if ents:
                st.caption("Mandatory entities (constraints):")
                tags = "".join(f'<span class="ent-tag">{e}</span>' for e in ents)
                st.markdown(tags, unsafe_allow_html=True)

        with t2:
            ent_data = [
                {"Entity": e, "Category": c}
                for c, items in pre["entities"].items()
                for e in items
            ]
            if ent_data:
                import pandas as pd
                st.dataframe(pd.DataFrame(ent_data), use_container_width=True, hide_index=True)

        with t3:
            avs = pre["action_sequence"]
            if avs:
                import pandas as pd
                st.dataframe(pd.DataFrame([{
                    "Verb": a["verb"],
                    "Subject": a["subject"] or "—",
                    "Object": a["object"] or "—",
                } for a in avs]), use_container_width=True, hide_index=True)

        with t4:
            ti = pre["temporal_info"]
            if ti:
                import pandas as pd
                st.dataframe(pd.DataFrame([{
                    "Marker": t["marker"],
                    "Sentence": t["sentence_idx"] + 1
                } for t in ti]), use_container_width=True, hide_index=True)
            else:
                st.caption("No temporal markers detected.")

    constraints = pre["constraints"]
    mandatory_entities = constraints["mandatory_entities"]

    # ── FASE 3 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">03 · LLM Conversion</p>', unsafe_allow_html=True)

    with st.spinner(f"Querying {len(active_models)} model(s)..."):
        t1 = time.time()
        llm_res = run_all_models(
            text=article_text,
            constraints=constraints,
            groq_key=groq_key,
            openai_key=openai_key,
            anthropic_key=anthropic_key
        )
        llm_time = time.time() - t1

    st.caption(f"Completed in {llm_time:.2f}s")

    # ── FASE 4 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">04 · Post-processing</p>', unsafe_allow_html=True)

    with st.spinner("Validating entities, numbering, format..."):
        post_res = {}
        for mk, lr in llm_res.items():
            if lr["success"]:
                post_res[mk] = run_postprocessing(lr["output"], mandatory_entities)
            else:
                post_res[mk] = {
                    "steps_numbered": [], "step_count": 0,
                    "entity_pass_rate": 0, "quality_score": {"total": 0},
                    "entities_missing": mandatory_entities,
                    "entities_found": [], "validated_text": "",
                    "corrections": [f"LLM error: {lr.get('error', 'unknown')}"]
                }

    # ── FASE 5 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">05 · Evaluation</p>', unsafe_allow_html=True)

    with st.spinner("Computing ROUGE and similarity scores..."):
        eval_res = {}
        for mk in llm_res:
            validated = post_res[mk].get("validated_text", "")
            epr = post_res[mk].get("entity_pass_rate", 0)
            eval_res[mk] = run_evaluation(validated, article_text, epr)

    st.caption("Evaluation complete.")

    winners = get_winners(post_res, eval_res)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">Results — Model Comparison</p>', unsafe_allow_html=True)

    # Winner row
    if winners:
        wcols = st.columns(len(winners))
        for i, (metric, wm) in enumerate(winners.items()):
            with wcols[i]:
                st.markdown(f"""
                <div class="winner-card">
                    <div class="metric-name">{metric}</div>
                    <div class="winner-name">{model_label(wm)}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")

    # 3-column output
    model_keys = ["groq", "gpt", "claude"]
    cols = st.columns(3)

    for i, mk in enumerate(model_keys):
        with cols[i]:
            lr  = llm_res[mk]
            pr  = post_res[mk]
            er  = eval_res[mk]
            cls = mk

            st.markdown(
                f'<div class="model-header {cls}">{model_label(mk)}</div>',
                unsafe_allow_html=True
            )

            if not lr["success"]:
                st.markdown(
                    f'<div class="error-box">{pr["corrections"][0]}</div>',
                    unsafe_allow_html=True
                )
                continue

            st.caption(f"{lr['elapsed_seconds']}s · {pr.get('step_count', 0)} steps")

            ep = pr.get("entity_pass_rate", 0)
            rl = er.get("rouge", {}).get("rougeL_f", 0) * 100
            bs = er.get("bertscore", {}).get("f1", 0) * 100
            qs = pr.get("quality_score", {}).get("total", 0)

            chips = (
                chip("Entity", ep, "%") + " " +
                chip("ROUGE-L", rl) + " " +
                chip("BERTScore", bs) + " " +
                chip("Quality", qs, "%")
            )
            st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

            validated = pr.get("validated_text", "")
            if validated:
                st.markdown(f'<div class="output-box">{validated}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="output-box" style="color:#aaa">No output.</div>', unsafe_allow_html=True)

            with st.expander("Entity validation"):
                found   = pr.get("entities_found", [])
                missing = pr.get("entities_missing", [])
                if found:
                    st.caption("Found")
                    for e in found:
                        st.markdown(f"- `{e}`")
                if missing:
                    st.caption("Missing")
                    for e in missing:
                        st.markdown(f"- `{e}`")

    # ── METRICS TABLE ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">Metrics Summary</p>', unsafe_allow_html=True)

    import pandas as pd
    rows = []
    for mk in model_keys:
        lr = llm_res[mk]
        pr = post_res[mk]
        er = eval_res[mk]
        rows.append({
            "Model":          model_label(mk),
            "Status":         "OK" if lr["success"] else "Error",
            "Steps":          pr.get("step_count", 0),
            "Latency (s)":    lr.get("elapsed_seconds", 0),
            "Entity (%)":     pr.get("entity_pass_rate", 0),
            "ROUGE-1":        round(er.get("rouge", {}).get("rouge1_f", 0), 4),
            "ROUGE-2":        round(er.get("rouge", {}).get("rouge2_f", 0), 4),
            "ROUGE-L":        round(er.get("rouge", {}).get("rougeL_f", 0), 4),
            "BERTScore P":    round(er.get("bertscore", {}).get("precision", 0), 4),
            "BERTScore R":    round(er.get("bertscore", {}).get("recall", 0), 4),
            "BERTScore F1":   round(er.get("bertscore", {}).get("f1", 0), 4),
            "Quality (%)":    pr.get("quality_score", {}).get("total", 0),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Entity (%)":  st.column_config.ProgressColumn("Entity (%)",  min_value=0, max_value=100),
            "Quality (%)": st.column_config.ProgressColumn("Quality (%)", min_value=0, max_value=100),
        }
    )

    # ── INSIGHTS ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">Insights</p>', unsafe_allow_html=True)

    ok = [mk for mk in model_keys if llm_res[mk]["success"]]
    if ok:
        ep_s    = {mk: post_res[mk].get("entity_pass_rate", 0) for mk in ok}
        rouge_s = {mk: eval_res[mk].get("rouge", {}).get("rougeL_f", 0) for mk in ok}
        bert_s  = {mk: eval_res[mk].get("bertscore", {}).get("f1", 0) for mk in ok}
        qual_s  = {mk: eval_res[mk].get("overall_quality", 0) for mk in ok}

        best_ep    = max(ep_s, key=ep_s.get)
        best_rouge = max(rouge_s, key=rouge_s.get)
        best_bert  = max(bert_s, key=bert_s.get)
        best_qual  = max(qual_s, key=qual_s.get)

        threshold_note = "meets the ≥85% research threshold" if ep_s[best_ep] >= 85 else "below the ≥85% research threshold"

        for text in [
            f"<b>Entity Preservation:</b> {model_label(best_ep)} leads at {ep_s[best_ep]:.1f}% — {threshold_note}.",
            f"<b>ROUGE-L:</b> {model_label(best_rouge)} achieves the highest lexical overlap ({rouge_s[best_rouge]:.4f}), indicating stronger content preservation.",
            f"<b>BERTScore F1:</b> {model_label(best_bert)} scores highest on semantic similarity ({bert_s[best_bert]:.4f}).",
            f"<b>Overall:</b> {model_label(best_qual)} performs best across combined metrics ({qual_s[best_qual]:.1f}%). Optimal model selection depends on which metric aligns with the research priority."
        ]:
            st.markdown(f'<div class="insight-row">{text}</div>', unsafe_allow_html=True)

    # ── EXPORT ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)

    ecols = st.columns(len(ok) + 1) if ok else st.columns(1)
    for i, mk in enumerate(ok):
        with ecols[i]:
            validated = post_res[mk].get("validated_text", "")
            if validated:
                st.download_button(
                    label=f"Download {mk.upper()} output (.txt)",
                    data=validated,
                    file_name=f"procedural_{mk}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    if ok:
        with ecols[-1]:
            st.download_button(
                label="Download metrics (.csv)",
                data=df.to_csv(index=False),
                file_name="comparison_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

# ── EMPTY STATE ───────────────────────────────────────────────────────────────
elif not run_btn:
    st.markdown("""
    <div style="padding: 3rem 0; text-align: center; color: #bbb;">
        <p style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; letter-spacing:0.05em;">
            ADD API KEY · PASTE ARTICLE · RUN PIPELINE
        </p>
    </div>
    """, unsafe_allow_html=True)