"""
Hybrid Pipeline Demo — Streamlit App
Lathifah Sahda · NRP 5025221159 · Teknik Informatika ITS

Tampilan komparatif: Groq | GPT | Claude
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Import pipeline modules
from pipeline.preprocessor import run_preprocessing
from pipeline.llm_converter import run_all_models
from pipeline.postprocessor import run_postprocessing
from pipeline.evaluator import run_evaluation

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid Pipeline Demo · ITS TA",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid #e94560;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #a0aec0;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }

    /* Phase badge */
    .phase-badge {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Model card */
    .model-card {
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #1a202c;
    }
    .model-card.groq { border-top: 3px solid #f6ad55; }
    .model-card.gpt  { border-top: 3px solid #68d391; }
    .model-card.claude { border-top: 3px solid #76e4f7; }

    /* Score pill */
    .score-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .score-high { background: #276749; color: #9ae6b4; }
    .score-mid  { background: #744210; color: #fbd38d; }
    .score-low  { background: #63171b; color: #fed7d7; }

    /* Winner banner */
    .winner-banner {
        background: linear-gradient(90deg, #1a3a1a, #276749);
        border: 1px solid #68d391;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        color: #9ae6b4;
        font-size: 0.85rem;
    }

    /* Step output */
    .step-output {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
        color: #c9d1d9;
        white-space: pre-wrap;
        max-height: 350px;
        overflow-y: auto;
    }

    /* Metric row */
    .metric-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.5rem 0;
    }

    /* Pipeline flow */
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        background: #2d3748;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 0.2rem;
        color: #e2e8f0;
    }
    .pipeline-arrow {
        color: #e94560;
        font-size: 1.2rem;
        margin: 0 0.2rem;
    }

    /* Sidebar */
    .sidebar-section {
        background: #1a202c;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border: 1px solid #2d3748;
    }

    /* Error box */
    .error-box {
        background: #2d1515;
        border: 1px solid #e53e3e;
        border-radius: 8px;
        padding: 0.8rem;
        color: #fc8181;
        font-size: 0.85rem;
    }

    /* Entity tag */
    .entity-tag {
        display: inline-block;
        background: #2a4365;
        color: #90cdf4;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.1rem;
    }
    .entity-tag.person { background: #44337a; color: #d6bcfa; }
    .entity-tag.org    { background: #1a365d; color: #90cdf4; }
    .entity-tag.date   { background: #1c4532; color: #9ae6b4; }
    .entity-tag.loc    { background: #2d3748; color: #e2e8f0; }
    .entity-tag.money  { background: #744210; color: #fbd38d; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def score_color_class(score: float) -> str:
    if score >= 70:
        return "score-high"
    elif score >= 40:
        return "score-mid"
    else:
        return "score-low"


def render_score_pill(label: str, value, suffix: str = "") -> str:
    try:
        v = float(value)
        cls = score_color_class(v)
        display = f"{v:.1f}{suffix}"
    except Exception:
        cls = "score-mid"
        display = str(value)
    return f'<span class="score-pill {cls}">{label}: {display}</span>'


def render_entity_tags(entities: dict) -> str:
    html = ""
    colors = {
        "PERSON": "person", "ORG": "org", "GPE": "loc",
        "LOC": "loc", "DATE": "date", "MONEY": "money", "MISC": ""
    }
    for cat, items in entities.items():
        for ent in items[:3]:  # max 3 per kategori
            cls = colors.get(cat, "")
            html += f'<span class="entity-tag {cls}">{ent} <small>({cat})</small></span>'
    return html if html else "<em style='color:#718096'>No entities found</em>"


def get_winner_analysis(results_post: dict, results_eval: dict) -> dict:
    """Tentukan pemenang per metrik."""
    winners = {}
    models = ["groq", "gpt", "claude"]

    # Entity Preservation
    ep = {m: results_post[m].get("entity_pass_rate", 0) for m in models if m in results_post}
    if ep:
        winners["entity"] = max(ep, key=ep.get)

    # ROUGE-L
    rouge = {
        m: results_eval[m].get("rouge", {}).get("rougeL_f", 0) * 100
        for m in models if m in results_eval
    }
    if rouge:
        winners["rouge_l"] = max(rouge, key=rouge.get)

    # BERTScore F1
    bert = {
        m: results_eval[m].get("bertscore", {}).get("f1", 0) * 100
        for m in models if m in results_eval
    }
    if bert:
        winners["bertscore"] = max(bert, key=bert.get)

    # Overall quality
    overall = {m: results_eval[m].get("overall_quality", 0) for m in models if m in results_eval}
    if overall:
        winners["overall"] = max(overall, key=overall.get)

    return winners


MODEL_COLORS = {
    "groq": "#f6ad55",
    "gpt": "#68d391",
    "claude": "#76e4f7"
}

MODEL_LABELS = {
    "groq": "🟠 Groq · Llama 3.3 70B",
    "gpt": "🟢 GPT · gpt-4o-mini",
    "claude": "🔵 Claude · Haiku"
}

# Artikel contoh
SAMPLE_ARTICLE = """Apple Inc. announced on Tuesday the launch of its new iPhone 16 series, 
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


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🔑 API Keys")
    st.caption("Keys tersimpan hanya di sesi ini, tidak disimpan.")

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Dapatkan gratis di console.groq.com"
    )
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="platform.openai.com/api-keys"
    )
    anthropic_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="console.anthropic.com"
    )

    st.divider()

    st.markdown("### 📊 Models Active")
    active_models = []
    if groq_key:
        st.success("🟠 Groq ✓")
        active_models.append("groq")
    else:
        st.error("🟠 Groq ✗ (no key)")

    if openai_key:
        st.success("🟢 GPT ✓")
        active_models.append("gpt")
    else:
        st.error("🟢 GPT ✗ (no key)")

    if anthropic_key:
        st.success("🔵 Claude ✓")
        active_models.append("claude")
    else:
        st.error("🔵 Claude ✗ (no key)")

    st.divider()
    st.markdown("### 📖 About")
    st.caption(
        "**Tugas Akhir**\n\n"
        "Lathifah Sahda · 5025221159\n\n"
        "Teknik Informatika ITS · 2026\n\n"
        "Pembimbing: Shintami Chusnul Hidayati, S.Kom., M.Sc., Ph.D."
    )


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>📰 Hybrid Pipeline: Narrative → Procedural</h1>
    <p>Perbandingan komparatif Groq · GPT · Claude dengan constraint-based LLM approach</p>
    <p style="font-size:0.8rem; margin-top:0.3rem; color:#718096">
        Tugas Akhir · Lathifah Sahda · NRP 5025221159 · Teknik Informatika ITS · 2026
    </p>
</div>
""", unsafe_allow_html=True)

# Pipeline flow indicator
st.markdown("""
<div style="text-align:center; margin-bottom:1.5rem;">
    <span class="pipeline-step">📥 Input Artikel</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">🔧 Fase 2: Pre-processing</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">🤖 Fase 3: LLM Conversion</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">✅ Fase 4: Post-processing</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">📊 Fase 5: Evaluasi</span>
</div>
""", unsafe_allow_html=True)

# ── INPUT SECTION ──
st.markdown("## 📥 Input Artikel Berita")

col_input, col_info = st.columns([3, 1])

with col_input:
    article_text = st.text_area(
        "Masukkan artikel berita (Bahasa Inggris, 200–1000 kata):",
        value=SAMPLE_ARTICLE,
        height=250,
        help="Artikel berita dalam bahasa Inggris. Disarankan domain: bisnis, teknologi, atau panduan praktis."
    )

with col_info:
    if article_text:
        wc = len(article_text.split())
        if 200 <= wc <= 1000:
            st.success(f"✅ {wc} kata\n\nDalam rentang yang valid (200–1000)")
        elif wc < 200:
            st.warning(f"⚠️ {wc} kata\n\nTerlalu pendek (min 200)")
        else:
            st.warning(f"⚠️ {wc} kata\n\nTerlalu panjang (max 1000)")

    st.info("💡 **Tips**\n\nArtikel dengan struktur kronologis yang jelas akan menghasilkan output prosedural terbaik.")

    if st.button("📋 Load Sample Article", use_container_width=True):
        st.rerun()

# Run button
st.markdown("")
run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    run_button = st.button(
        "🚀 Jalankan Pipeline (Semua Model)",
        type="primary",
        use_container_width=True,
        disabled=len(active_models) == 0
    )

if len(active_models) == 0:
    st.warning("⚠️ Masukkan minimal satu API key di sidebar untuk menjalankan pipeline.")

# ─────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────

if run_button and article_text and len(active_models) > 0:

    # ── FASE 2: PRE-PROCESSING ──
    st.markdown("---")
    st.markdown("## 🔧 Fase 2: Rule-Based Pre-Processing")

    with st.spinner("Memproses teks... (NER, action verbs, temporal markers)"):
        t0 = time.time()
        pre_result = run_preprocessing(article_text)
        pre_time = time.time() - t0

    st.success(f"✅ Pre-processing selesai dalam {pre_time:.2f}s")

    # Tampilkan hasil pre-processing
    with st.expander("📋 Lihat Detail Pre-Processing", expanded=False):

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🏷️ Entities", "⚡ Action Verbs", "⏰ Temporal"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Word Count", pre_result["word_count"])
            c2.metric("Sentences", pre_result["sentence_count"])
            c3.metric("Action Verbs", len(pre_result["action_sequence"]))

            st.markdown("**Generated Constraints:**")
            constraints = pre_result["constraints"]
            if constraints["mandatory_entities"]:
                st.markdown("*Mandatory Entities:*")
                ent_html = "".join(
                    f'<span class="entity-tag">{e}</span>'
                    for e in constraints["mandatory_entities"]
                )
                st.markdown(ent_html, unsafe_allow_html=True)

        with tab2:
            entities = pre_result["entities"]
            st.markdown(render_entity_tags(entities), unsafe_allow_html=True)

            # Tabel entities
            ent_data = []
            for cat, items in entities.items():
                for ent in items:
                    ent_data.append({"Entity": ent, "Category": cat})
            if ent_data:
                import pandas as pd
                st.dataframe(pd.DataFrame(ent_data), use_container_width=True, hide_index=True)

        with tab3:
            avs = pre_result["action_sequence"]
            if avs:
                import pandas as pd
                df_av = pd.DataFrame([
                    {
                        "Verb": av["verb"],
                        "Subject": av["subject"] or "-",
                        "Object": av["object"] or "-",
                        "Sentence": av["sentence"][:80] + "..."
                    }
                    for av in avs
                ])
                st.dataframe(df_av, use_container_width=True, hide_index=True)
            else:
                st.info("No action verbs detected.")

        with tab4:
            ti = pre_result["temporal_info"]
            if ti:
                import pandas as pd
                df_ti = pd.DataFrame([
                    {"Marker": t["marker"], "Sentence #": t["sentence_idx"] + 1}
                    for t in ti
                ])
                st.dataframe(df_ti, use_container_width=True, hide_index=True)
            else:
                st.info("No temporal markers detected.")

    constraints = pre_result["constraints"]
    mandatory_entities = constraints["mandatory_entities"]

    # ── FASE 3: LLM CONVERSION ──
    st.markdown("---")
    st.markdown("## 🤖 Fase 3: Constrained LLM Conversion")

    with st.spinner(f"Menjalankan {len(active_models)} model secara paralel..."):
        t1 = time.time()
        llm_results = run_all_models(
            text=article_text,
            constraints=constraints,
            groq_key=groq_key,
            openai_key=openai_key,
            anthropic_key=anthropic_key
        )
        llm_time = time.time() - t1

    st.success(f"✅ LLM conversion selesai dalam {llm_time:.2f}s")

    # ── FASE 4: POST-PROCESSING ──
    st.markdown("---")
    st.markdown("## ✅ Fase 4: Rule-Based Post-Processing")

    with st.spinner("Validasi dan standarisasi output..."):
        post_results = {}
        for model_key, llm_res in llm_results.items():
            if llm_res["success"]:
                post_results[model_key] = run_postprocessing(
                    llm_res["output"],
                    mandatory_entities
                )
            else:
                post_results[model_key] = {
                    "steps_numbered": [],
                    "step_count": 0,
                    "entity_pass_rate": 0,
                    "quality_score": {"total": 0},
                    "entities_missing": mandatory_entities,
                    "entities_found": [],
                    "validated_text": "",
                    "corrections": [f"LLM failed: {llm_res.get('error', 'Unknown')}"]
                }

    # ── FASE 5: EVALUASI ──
    st.markdown("---")
    st.markdown("## 📊 Fase 5: Evaluasi Otomatis")

    with st.spinner("Menghitung ROUGE + BERTScore..."):
        eval_results = {}
        for model_key in llm_results:
            validated = post_results[model_key].get("validated_text", "")
            epr = post_results[model_key].get("entity_pass_rate", 0)
            eval_results[model_key] = run_evaluation(validated, article_text, epr)

    st.success("✅ Evaluasi selesai!")

    # ── WINNER ANALYSIS ──
    winners = get_winner_analysis(post_results, eval_results)

    # ─────────────────────────────────────────────
    # KOMPARATIF OUTPUT
    # ─────────────────────────────────────────────

    st.markdown("---")
    st.markdown("## 🏆 Perbandingan Hasil Ketiga Model")

    # Winner summary
    if winners:
        st.markdown("### 🥇 Ringkasan Pemenang")
        win_cols = st.columns(len(winners))
        winner_labels = {
            "entity": "Entity Preservation",
            "rouge_l": "ROUGE-L",
            "bertscore": "BERTScore F1",
            "overall": "Overall Quality"
        }
        icons = {"entity": "🏷️", "rouge_l": "📝", "bertscore": "🧠", "overall": "⭐"}

        for i, (metric, winner_model) in enumerate(winners.items()):
            with win_cols[i % len(win_cols)]:
                color = MODEL_COLORS.get(winner_model, "#fff")
                label = MODEL_LABELS.get(winner_model, winner_model)
                st.markdown(f"""
                <div class="winner-banner">
                    {icons.get(metric, '🏆')} <strong>{winner_labels.get(metric, metric)}</strong><br>
                    <span style="color:{color}; font-size:1.1rem;">▶ {label}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── 3 KOLOM SIDE BY SIDE ──
    st.markdown("### 📋 Output Prosedural")

    cols = st.columns(3)
    model_keys = ["groq", "gpt", "claude"]

    for i, model_key in enumerate(model_keys):
        with cols[i]:
            llm_res = llm_results[model_key]
            post_res = post_results[model_key]
            eval_res = eval_results[model_key]
            color = MODEL_COLORS[model_key]
            label = MODEL_LABELS[model_key]

            # Model header
            is_winner = model_key in winners.values()
            winner_badge = " 🏆" if is_winner else ""
            st.markdown(
                f'<div style="color:{color}; font-size:1.1rem; font-weight:700; '
                f'border-bottom: 2px solid {color}; padding-bottom:0.5rem; margin-bottom:0.8rem;">'
                f'{label}{winner_badge}</div>',
                unsafe_allow_html=True
            )

            if not llm_res["success"]:
                st.markdown(
                    f'<div class="error-box">❌ Error: {llm_res.get("error", "Unknown error")}</div>',
                    unsafe_allow_html=True
                )
                continue

            # Timing
            st.caption(f"⏱️ {llm_res['elapsed_seconds']}s · {post_res.get('step_count', 0)} steps")

            # Metrics row
            ep = post_res.get("entity_pass_rate", 0)
            rl = eval_res.get("rouge", {}).get("rougeL_f", 0) * 100
            bs = eval_res.get("bertscore", {}).get("f1", 0) * 100
            qs = post_res.get("quality_score", {}).get("total", 0)

            pills = (
                render_score_pill("Entity", ep, "%") + " " +
                render_score_pill("ROUGE-L", rl, "") + " " +
                render_score_pill("BERTScore", bs, "") + " " +
                render_score_pill("Quality", qs, "%")
            )
            st.markdown(f'<div class="metric-row">{pills}</div>', unsafe_allow_html=True)

            # Output text
            validated = post_res.get("validated_text", "")
            if validated:
                st.markdown(
                    f'<div class="step-output">{validated}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="step-output" style="color:#718096">No output generated.</div>',
                    unsafe_allow_html=True
                )

            # Entity status
            with st.expander("🏷️ Entity Validation"):
                found = post_res.get("entities_found", [])
                missing = post_res.get("entities_missing", [])
                if found:
                    st.markdown("**Found ✅**")
                    for e in found:
                        st.markdown(f"- `{e}`")
                if missing:
                    st.markdown("**Missing ❌**")
                    for e in missing:
                        st.markdown(f"- `{e}`")

    # ── TABEL PERBANDINGAN METRIK ──
    st.markdown("---")
    st.markdown("### 📊 Tabel Perbandingan Metrik Lengkap")

    import pandas as pd

    table_data = []
    for model_key in model_keys:
        llm_res = llm_results[model_key]
        post_res = post_results[model_key]
        eval_res = eval_results[model_key]

        table_data.append({
            "Model": MODEL_LABELS[model_key],
            "Status": "✅" if llm_res["success"] else "❌",
            "Steps": post_res.get("step_count", 0),
            "Latency (s)": llm_res.get("elapsed_seconds", 0),
            "Entity Pass (%)": post_res.get("entity_pass_rate", 0),
            "ROUGE-1": round(eval_res.get("rouge", {}).get("rouge1_f", 0), 4),
            "ROUGE-2": round(eval_res.get("rouge", {}).get("rouge2_f", 0), 4),
            "ROUGE-L": round(eval_res.get("rouge", {}).get("rougeL_f", 0), 4),
            "BERTScore P": round(eval_res.get("bertscore", {}).get("precision", 0), 4),
            "BERTScore R": round(eval_res.get("bertscore", {}).get("recall", 0), 4),
            "BERTScore F1": round(eval_res.get("bertscore", {}).get("f1", 0), 4),
            "Quality Score (%)": post_res.get("quality_score", {}).get("total", 0),
        })

    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entity Pass (%)": st.column_config.ProgressColumn(
                "Entity Pass (%)", min_value=0, max_value=100
            ),
            "Quality Score (%)": st.column_config.ProgressColumn(
                "Quality Score (%)", min_value=0, max_value=100
            ),
        }
    )

    # ── INSIGHTS ──
    st.markdown("---")
    st.markdown("### 💡 Insights Otomatis")

    successful = [k for k in model_keys if llm_results[k]["success"]]
    if successful:
        insights = []

        # Entity preservation insight
        ep_scores = {k: post_results[k].get("entity_pass_rate", 0) for k in successful}
        best_ep = max(ep_scores, key=ep_scores.get)
        insights.append(
            f"**Entity Preservation:** {MODEL_LABELS[best_ep]} unggul dengan "
            f"{ep_scores[best_ep]:.1f}% entity pass rate — "
            f"{'sangat baik' if ep_scores[best_ep] >= 85 else 'perlu ditingkatkan'} "
            f"(threshold penelitian ≥85%)."
        )

        # ROUGE insight
        rouge_scores = {
            k: eval_results[k].get("rouge", {}).get("rougeL_f", 0)
            for k in successful
        }
        best_rouge = max(rouge_scores, key=rouge_scores.get)
        insights.append(
            f"**ROUGE-L:** {MODEL_LABELS[best_rouge]} menghasilkan skor tertinggi "
            f"({rouge_scores[best_rouge]:.4f}) — menunjukkan preservasi konten leksikal terbaik."
        )

        # BERTScore insight
        bert_scores = {
            k: eval_results[k].get("bertscore", {}).get("f1", 0)
            for k in successful
        }
        best_bert = max(bert_scores, key=bert_scores.get)
        insights.append(
            f"**BERTScore F1:** {MODEL_LABELS[best_bert]} unggul "
            f"({bert_scores[best_bert]:.4f}) — kesamaan semantik terbaik dengan artikel asli."
        )

        # Recommendation
        overall_scores = {k: eval_results[k].get("overall_quality", 0) for k in successful}
        best_overall = max(overall_scores, key=overall_scores.get)
        insights.append(
            f"**Rekomendasi:** Berdasarkan kombinasi semua metrik, "
            f"{MODEL_LABELS[best_overall]} memberikan performa terbaik secara keseluruhan. "
            f"Namun pemilihan model optimal bergantung pada prioritas metrik penelitian."
        )

        for insight in insights:
            st.info(insight)

    # ── DOWNLOAD ──
    st.markdown("---")
    st.markdown("### 💾 Export Hasil")

    dl_cols = st.columns(len(successful)) if successful else st.columns(1)

    for i, model_key in enumerate(successful):
        with dl_cols[i]:
            validated = post_results[model_key].get("validated_text", "")
            if validated:
                st.download_button(
                    label=f"⬇️ Download {model_key.upper()} output",
                    data=validated,
                    file_name=f"procedural_{model_key}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Export tabel CSV
    if table_data:
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Comparison Table (CSV)",
            data=csv,
            file_name="comparison_metrics.csv",
            mime="text/csv"
        )

# ── EMPTY STATE ──
elif not run_button:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #718096;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📰 → 📋</div>
        <h3 style="color:#a0aec0;">Siap untuk dijalankan</h3>
        <p>Masukkan API key di sidebar, lalu klik <strong>Jalankan Pipeline</strong></p>
        <p style="font-size:0.85rem; margin-top:1rem;">
            Pipeline akan memproses artikel melalui 5 fase:<br>
            Pre-processing → LLM Conversion (3 model) → Post-processing → Evaluasi
        </p>
    </div>
    """, unsafe_allow_html=True)
