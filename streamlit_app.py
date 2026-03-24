"""OpenClaw Memory Pro System Dashboard — v0.0.6"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

MEMORY_SERVER = os.environ.get("MEMORY_SERVER_URL", "http://127.0.0.1:18790")
WORKSPACE = Path(os.environ.get("OPENCLAW_WORKSPACE", Path(__file__).parent))
CONTEXT_FILES = {
    "Memory": WORKSPACE / "MEMORY.md",
    "Agents": WORKSPACE / "AGENTS.md",
    "User": WORKSPACE / "USER.md",
    "Soul": WORKSPACE / "SOUL.md",
}

st.set_page_config(page_title="Memory Pro System", layout="wide", page_icon="🧠")

# ── Design System ────────────────────────────────────────────────

COLORS = {
    "bg": "#0e1117",
    "card": "#1a1d23",
    "card_hover": "#22262e",
    "border": "#2d333b",
    "text": "#e6edf3",
    "text_dim": "#8b949e",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "yellow": "#d29922",
    "red": "#f85149",
    "purple": "#bc8cff",
    "orange": "#f0883e",
    "cyan": "#39d2c0",
    "pink": "#f778ba",
    "fact": "#3fb950",
    "decision": "#58a6ff",
    "preference": "#f0883e",
    "goal": "#bc8cff",
    "question": "#f85149",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{ font-family: 'Inter', sans-serif; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #13161b 0%, #0e1117 100%);
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }}

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {COLORS['card']};
        border-radius: 12px;
        padding: 4px;
        border: 1px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 0.01em;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['accent']}22;
        color: {COLORS['accent']} !important;
    }}

    /* Metric card */
    .metric-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 20px 24px;
        transition: all 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: {COLORS['accent']}44;
        box-shadow: 0 4px 24px rgba(88,166,255,0.08);
    }}
    .metric-label {{
        font-size: 0.75rem;
        font-weight: 500;
        color: {COLORS['text_dim']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['text']};
        line-height: 1.1;
    }}
    .metric-delta {{
        font-size: 0.8rem;
        color: {COLORS['text_dim']};
        margin-top: 4px;
    }}
    .metric-green .metric-value {{ color: {COLORS['green']}; }}
    .metric-blue .metric-value {{ color: {COLORS['accent']}; }}
    .metric-purple .metric-value {{ color: {COLORS['purple']}; }}
    .metric-orange .metric-value {{ color: {COLORS['orange']}; }}

    /* Section card */
    .section-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }}
    .section-title {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {COLORS['text_dim']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-title .dot {{
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }}

    /* Skill card */
    .skill-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
    }}
    .skill-card:hover {{
        border-color: {COLORS['accent']}66;
        background: {COLORS['card_hover']};
    }}
    .skill-card.active {{
        border-color: {COLORS['accent']};
        background: {COLORS['accent']}0a;
    }}
    .skill-name {{
        font-weight: 600;
        font-size: 0.95rem;
        color: {COLORS['text']};
    }}
    .skill-meta {{
        font-size: 0.75rem;
        color: {COLORS['text_dim']};
        margin-top: 2px;
    }}

    /* Utility bar */
    .util-bar-bg {{
        background: {COLORS['border']};
        border-radius: 4px;
        height: 6px;
        margin-top: 8px;
        overflow: hidden;
    }}
    .util-bar-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }}
    .util-high {{ background: {COLORS['green']}; }}
    .util-mid {{ background: {COLORS['yellow']}; }}
    .util-low {{ background: {COLORS['red']}; }}

    /* Badge */
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-active {{ background: {COLORS['green']}22; color: {COLORS['green']}; }}
    .badge-draft {{ background: {COLORS['yellow']}22; color: {COLORS['yellow']}; }}
    .badge-deprecated {{ background: {COLORS['red']}22; color: {COLORS['red']}; }}

    /* KG legend */
    .kg-legend {{
        display: flex;
        gap: 16px;
        padding: 12px 16px;
        background: {COLORS['card']};
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        margin-top: 8px;
    }}
    .kg-legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        color: {COLORS['text_dim']};
    }}
    .kg-dot {{
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
    }}

    /* Pipeline diagram */
    .pipeline {{
        background: linear-gradient(135deg, {COLORS['card']} 0%, #1e2229 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 24px;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.78rem;
        line-height: 1.7;
        color: {COLORS['text_dim']};
        overflow-x: auto;
    }}
    .pipeline .hl {{ color: {COLORS['accent']}; font-weight: 600; }}
    .pipeline .hl-green {{ color: {COLORS['green']}; }}
    .pipeline .hl-purple {{ color: {COLORS['purple']}; }}
    .pipeline .hl-orange {{ color: {COLORS['orange']}; }}
    .pipeline .hl-cyan {{ color: {COLORS['cyan']}; }}
    .pipeline .dim {{ color: #484f58; }}

    /* Briefing box */
    .briefing {{
        background: linear-gradient(135deg, {COLORS['accent']}08 0%, {COLORS['purple']}08 100%);
        border: 1px solid {COLORS['accent']}22;
        border-radius: 12px;
        padding: 20px 24px;
        font-size: 0.9rem;
        line-height: 1.6;
        color: {COLORS['text']};
    }}

    /* Insight card */
    .insight-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }}
    .insight-card .novelty {{
        font-size: 1.4rem;
        font-weight: 700;
    }}

    /* Editor area */
    .stTextArea textarea {{
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important;
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
    }}

    /* Status dot */
    .status-online {{ color: {COLORS['green']}; }}
    .status-offline {{ color: {COLORS['red']}; }}

    /* Override some Streamlit defaults */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 6px 16px;
        border: 1px solid {COLORS['border']};
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {COLORS['accent']};
        color: {COLORS['accent']};
    }}
    div[data-testid="stExpander"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
    }}
    .stDivider {{
        border-color: {COLORS['border']} !important;
    }}

    /* Contradiction / blindspot cards */
    .inference-card {{
        background: {COLORS['card']};
        border-left: 3px solid;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin-bottom: 10px;
    }}
    .inference-card.contradiction {{ border-left-color: {COLORS['red']}; }}
    .inference-card.blindspot {{ border-left-color: {COLORS['yellow']}; }}
    .inference-card.thread {{ border-left-color: {COLORS['purple']}; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────

def api_get(path: str, timeout: float = 10):
    try:
        r = requests.get(f"{MEMORY_SERVER}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, body: dict = None, timeout: float = 30):
    try:
        r = requests.post(f"{MEMORY_SERVER}{path}", json=body or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def server_online() -> bool:
    h = api_get("/health", timeout=3)
    return h is not None and h.get("status") == "ok"


def fmt_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16] if ts else ""


def read_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def metric_card(label: str, value, delta: str = "", color: str = "blue"):
    st.markdown(f"""
    <div class="metric-card metric-{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>""", unsafe_allow_html=True)


def util_bar(ratio: float) -> str:
    pct = max(0, min(100, int(ratio * 100)))
    cls = "util-high" if ratio >= 0.6 else ("util-mid" if ratio >= 0.3 else "util-low")
    return (f'<div class="util-bar-bg">'
            f'<div class="util-bar-fill {cls}" style="width:{pct}%"></div></div>')


def badge(status: str) -> str:
    return f'<span class="badge badge-{status}">{status}</span>'


# ── Sidebar ──────────────────────────────────────────────────────

online = server_online()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 8px;">
        <div style="font-size:2rem;">🧠</div>
        <div style="font-size:1rem; font-weight:700; letter-spacing:0.04em; margin-top:4px;">
            MEMORY PRO
        </div>
        <div style="font-size:0.7rem; color:#8b949e; letter-spacing:0.1em;">
            SYSTEM v0.0.6
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if online:
        health = api_get("/health") or {}
        mins = health.get("uptime_seconds", 0) // 60
        hrs = mins // 60
        uptime_str = f"{hrs}h {mins % 60}m" if hrs else f"{mins}m"
        st.markdown(f"""
        <div style="padding:8px 0;">
            <span class="status-online" style="font-size:1.4rem;">●</span>
            <span style="font-weight:600; margin-left:6px;">Online</span>
            <span style="color:#8b949e; font-size:0.8rem; margin-left:8px;">{uptime_str}</span>
        </div>
        <div style="font-size:0.75rem; color:#8b949e; padding-left:24px;">
            PID {health.get('pid', '?')} · {health.get('embedder', '?')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:8px 0;">
            <span class="status-offline" style="font-size:1.4rem;">●</span>
            <span style="font-weight:600; margin-left:6px;">Offline</span>
        </div>
        <div style="font-size:0.75rem; color:#8b949e; padding-left:24px;">
            <code>memory-cli server-start</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Quick Actions</div>', unsafe_allow_html=True)

    with st.expander("💾 Remember", expanded=False):
        _c = st.text_area("Content", key="sb_r", height=60, placeholder="Quick memory...",
                          label_visibility="collapsed")
        _imp = st.slider("Importance", 0.0, 1.0, 0.7, 0.05, key="sb_i")
        if st.button("Save", key="sb_sv", use_container_width=True) and _c.strip():
            r = api_post("/remember", {"content": _c.strip(), "importance": _imp})
            st.toast("Saved" if r else "Failed")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ Collide", key="sb_co", use_container_width=True):
            api_post("/second-brain/collide", {"async": True})
            st.toast("Collision started")
    with c2:
        if st.button("📋 Digest", key="sb_di", use_container_width=True):
            api_post("/digest", {"days": 1, "async": True})
            st.toast("Digest started")

    if st.button("🎯 Propose Skill", key="sb_pr", use_container_width=True):
        r = api_post("/skills/propose")
        ct = r.get("count", 0) if r else 0
        st.toast(f"{ct} proposed" if ct else "No proposals")

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; font-size:0.75rem; color:#484f58;">
        {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────

tabs = st.tabs(["◈ Overview", "⚙ Skills", "🔬 Second Brain",
                "📝 Memory", "🤖 Agents", "👤 User", "✦ Soul"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    if not online:
        st.markdown("""
        <div class="briefing" style="text-align:center; padding:60px 24px;">
            <div style="font-size:3rem; margin-bottom:16px;">🧠</div>
            <div style="font-size:1.1rem; font-weight:600;">Memory Server Offline</div>
            <div style="color:#8b949e; margin-top:8px;">
                Start with <code>memory-cli server-start</code>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        status = api_get("/status") or {}
        kg_st = api_get("/kg/status") or {}
        sk_st = api_get("/skills/stats") or {}
        in_st = api_get("/insight/stats") or {}

        systems = status.get("systems", {})
        memora_info = systems.get("memora", {})
        mem_count = memora_info.get("count", memora_info.get("total", 0))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Memories", mem_count, "total stored", "green")
        with c2:
            active = sk_st.get("active", 0)
            total = sk_st.get("total", 0)
            metric_card("Skills", f"{active}/{total}", "active / total", "blue")
        with c3:
            nn = kg_st.get("total_nodes", 0)
            ne = kg_st.get("total_edges", 0)
            metric_card("KG Nodes", nn, f"{ne} edges", "purple")
        with c4:
            tr = in_st.get("total_ratings", 0)
            ar = in_st.get("average_rating", 0)
            metric_card("Insights", tr, f"avg {ar:.1f}" if ar else "no ratings", "orange")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        left, right = st.columns([3, 2])

        with left:
            st.markdown("""
            <div class="pipeline">
<span class="hl">Fragments</span> <span class="dim">───→</span> <span class="hl-green">[Ingest + Tag]</span> <span class="dim">───→</span> <span class="hl">Unified Corpus</span>
<span class="dim">                                         │</span>
<span class="dim">                               ┌─────────┼──────────┐</span>
<span class="dim">                               ↓         ↓          ↓</span>
                          <span class="hl-green">[KG Weave]</span>  <span class="hl-orange">[Distill]</span>  <span class="hl-purple">[Collide]</span>
                          <span class="dim">structural  compression   novelty</span>
                          <span class="dim"> _gain       _value       (1-5)</span>
<span class="dim">                               │         │          │</span>
<span class="dim">                               └────┬────┼────┬─────┘</span>
<span class="dim">                                    ↓</span>
                           <span class="hl-cyan">[Skill Proposer]</span>  <span class="dim">← 2-of-3 pass</span>
<span class="dim">                                    ↓</span>
                            <span class="hl">[Skill Registry]</span>  <span class="dim">← utility + feedback</span>
<span class="dim">                                    │</span>
<span class="dim">                          ┌─────────┼──────────┐</span>
<span class="dim">                          ↓         ↓          ↓</span>
                     <span class="hl">[Recall]</span>   <span class="hl-orange">[Push]</span>    <span class="hl-purple">[Finetune]</span>
<span class="dim">                          ↓</span>
                     <span class="hl-cyan">[Feedback → Rewrite]</span>  <span class="dim">← closed loop</span>
            </div>""", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-title">'
                        '<span class="dot" style="background:#3fb950"></span>'
                        'Memory Vitality</div>', unsafe_allow_html=True)

            vitality = api_get("/vitality")
            if vitality and "distribution" in vitality:
                dist = vitality["distribution"]
                import plotly.graph_objects as go
                colors_v = [COLORS["green"], COLORS["accent"],
                            COLORS["yellow"], COLORS["red"]]
                labels = list(dist.keys())
                values = list(dist.values())
                fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=0.55,
                    textinfo="label+percent",
                    marker=dict(colors=colors_v[:len(labels)],
                                line=dict(color=COLORS["bg"], width=2)),
                    textfont=dict(size=11, family="Inter"),
                )])
                fig.update_layout(
                    margin=dict(t=5, b=5, l=5, r=5), height=220,
                    showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text_dim"]),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No vitality data")

            st.markdown('<div class="section-title">'
                        '<span class="dot" style="background:#f0883e"></span>'
                        'Strategy Weights</div>', unsafe_allow_html=True)

            if in_st and "weights" in in_st:
                weights = in_st["weights"]
                import plotly.express as px
                names = list(weights.keys())
                vals = [weights[n] for n in names]
                fig = px.bar(y=names, x=vals, orientation="h",
                             color_discrete_sequence=[COLORS["accent"]])
                fig.update_layout(
                    margin=dict(t=5, b=5, l=5, r=5), height=180,
                    showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text_dim"], size=11),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False),
                )
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No strategy data")

        briefing = api_get("/briefing")
        if briefing and "text" in briefing:
            st.markdown(f"""
            <div class="section-title" style="margin-top:8px;">
                <span class="dot" style="background:#58a6ff"></span>
                Daily Briefing
            </div>
            <div class="briefing">{briefing['text']}</div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: Skills
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    if not online:
        st.info("Server offline")
    else:
        skills_data = api_get("/skills") or {"skills": [], "count": 0}
        all_skills = skills_data.get("skills", [])

        hdr1, hdr2, hdr3 = st.columns([3, 1, 1])
        with hdr1:
            st.markdown(f"""
            <div style="font-size:1.3rem; font-weight:700;">
                Skills <span style="color:{COLORS['text_dim']}; font-weight:400;">
                ({len(all_skills)})</span>
            </div>""", unsafe_allow_html=True)
        with hdr2:
            if st.button("🎯 Propose", key="sk_p", use_container_width=True):
                r = api_post("/skills/propose")
                ct = r.get("count", 0) if r else 0
                st.toast(f"{ct} proposed" if ct else "No proposals")
        with hdr3:
            flt = st.selectbox("Filter", ["all", "active", "draft", "deprecated"],
                               key="sk_f", label_visibility="collapsed")

        filtered = all_skills if flt == "all" else [
            s for s in all_skills if s.get("status") == flt]

        left, right = st.columns([2, 3])

        with left:
            if not filtered:
                st.markdown(f"""
                <div class="section-card" style="text-align:center; padding:40px;">
                    <div style="font-size:2rem; margin-bottom:8px;">⚙</div>
                    <div style="color:{COLORS['text_dim']}">No skills yet</div>
                </div>""", unsafe_allow_html=True)

            for i, sk in enumerate(filtered):
                name = sk.get("name", "Unnamed")
                sts = sk.get("status", "?")
                ver = sk.get("version", 1)
                util = sk.get("utility_rate", 0.5)
                succ = sk.get("successes", 0)
                fail = sk.get("failures", 0)
                total_u = succ + fail

                util_cls = ("util-high" if util >= 0.6
                            else "util-mid" if util >= 0.3 else "util-low")
                warn = (' <span style="color:#f85149; font-size:0.7rem;">⚠ low</span>'
                        if util < 0.3 and total_u >= 3 else "")

                st.markdown(f"""
                <div class="skill-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="skill-name">{name}</span>
                        {badge(sts)}
                    </div>
                    <div class="skill-meta">v{ver} · {total_u} uses ({succ}✓ {fail}✗){warn}</div>
                    {util_bar(util)}
                </div>""", unsafe_allow_html=True)

                if st.button(f"View", key=f"sv_{i}",
                             use_container_width=True):
                    st.session_state["sel_skill"] = sk

        with right:
            sk = st.session_state.get("sel_skill",
                                       filtered[0] if filtered else None)
            if sk:
                st.markdown(f"""
                <div class="section-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:1.2rem; font-weight:700;">{sk.get('name', '')}</div>
                            <div style="color:{COLORS['text_dim']}; font-size:0.8rem; margin-top:2px;">
                                {badge(sk.get('status', '?'))}
                                <span style="margin-left:8px;">v{sk.get('version', 1)}</span>
                                <span style="margin-left:8px;">
                                    {', '.join(sk.get('tags', []))}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                util = sk.get("utility_rate", 0.5)
                succ = sk.get("successes", 0)
                fail = sk.get("failures", 0)
                st.markdown(f"""
                <div class="section-card">
                    <div class="section-title">Utility</div>
                    <div style="display:flex; align-items:baseline; gap:12px;">
                        <span style="font-size:2rem; font-weight:700;
                            color:{'#3fb950' if util >= 0.6 else '#d29922' if util >= 0.3 else '#f85149'};">
                            {util:.0%}
                        </span>
                        <span style="color:{COLORS['text_dim']}; font-size:0.85rem;">
                            {succ} success · {fail} failure
                        </span>
                    </div>
                    {util_bar(util)}
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class="section-card">
                    <div class="section-title">Content</div>
                    <div style="line-height:1.6;">{sk.get('content', '*No content*')}</div>
                </div>""", unsafe_allow_html=True)

                prereqs = sk.get("prerequisites", "")
                procs = sk.get("procedures", "")
                applicable = sk.get("applicable_scenarios", "")
                inapplicable = sk.get("inapplicable_scenarios", "")
                if any([prereqs, procs, applicable, inapplicable]):
                    parts = []
                    if prereqs:
                        parts.append(f"<b>Prerequisites:</b> {prereqs}")
                    if procs:
                        parts.append(f"<b>Procedures:</b> {procs}")
                    if applicable:
                        parts.append(f"<b>Applicable:</b> {applicable}")
                    if inapplicable:
                        parts.append(f"<b>Inapplicable:</b> {inapplicable}")
                    st.markdown(f"""<div class="section-card">
                        <div class="section-title">Structured Details</div>
                        {'<br>'.join(parts)}
                    </div>""", unsafe_allow_html=True)

                ac1, ac2 = st.columns(2)
                with ac1:
                    if sk.get("status") == "draft":
                        if st.button("✅ Promote", key="sk_pro",
                                     use_container_width=True):
                            api_post("/skills/promote", {"skill_id": sk["id"]})
                            st.rerun()
                    elif sk.get("status") == "active":
                        if st.button("🚫 Deprecate", key="sk_dep",
                                     use_container_width=True):
                            api_post("/skills/deprecate", {"skill_id": sk["id"]})
                            st.rerun()
                with ac2:
                    fb = st.selectbox("Outcome", ["success", "failure"], key="sk_fbo",
                                      label_visibility="collapsed")
                    if st.button("📊 Record", key="sk_fbr", use_container_width=True):
                        api_post("/skills/feedback", {
                            "skill_id": sk["id"], "outcome": fb, "query": "dashboard"})
                        st.rerun()

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="section-title">
            <span class="dot" style="background:{COLORS['accent']}"></span>
            Skill Manifests</div>""", unsafe_allow_html=True)

        skill_dirs = sorted((WORKSPACE / "skills").glob("*/SKILL.md"))
        if skill_dirs:
            mtabs = st.tabs([p.parent.name for p in skill_dirs])
            for mt, sp in zip(mtabs, skill_dirs):
                with mt:
                    st.markdown(read_md(sp))
        else:
            st.caption("No SKILL.md manifests in skills/")


# ═══════════════════════════════════════════════════════════════════
# TAB 3: Second Brain
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    if not online:
        st.info("Server offline")
    else:
        sb_tabs = st.tabs(["🕸 Graph", "⚠ Contradictions", "🔍 Blindspots",
                           "🧵 Threads", "💡 Insights", "📊 Report"])

        with sb_tabs[0]:
            graph_data = api_get("/kg/graph")
            kg_st = api_get("/kg/status") or {}

            if graph_data and graph_data.get("nodes"):
                try:
                    from streamlit_agraph import agraph, Node, Edge, Config

                    type_colors = {
                        "fact": COLORS["fact"],
                        "decision": COLORS["decision"],
                        "preference": COLORS["preference"],
                        "goal": COLORS["goal"],
                        "question": COLORS["question"],
                    }

                    ag_nodes = []
                    for n in graph_data["nodes"]:
                        lbl = n["content"][:35] + ("…" if len(n["content"]) > 35 else "")
                        ag_nodes.append(Node(
                            id=n["id"], label=lbl,
                            size=14 + n.get("importance", 0.5) * 22,
                            color=type_colors.get(n.get("node_type", ""), "#666"),
                            title=f"[{n.get('node_type', '?')}] {n['content']}",
                            font={"color": COLORS["text"], "size": 10},
                        ))

                    ag_edges = []
                    for e in graph_data["edges"]:
                        edge_color = (COLORS["red"] if e.get("edge_type") == "contradicts"
                                      else "#444")
                        ag_edges.append(Edge(
                            source=e["source_id"], target=e["target_id"],
                            label=e.get("edge_type", ""),
                            color=edge_color, width=1,
                        ))

                    config = Config(width=900, height=500, directed=True,
                                    physics=True, hierarchical=False,
                                    backgroundColor=COLORS["bg"])
                    agraph(nodes=ag_nodes, edges=ag_edges, config=config)
                except ImportError:
                    st.warning("Install `streamlit-agraph`: `pip install streamlit-agraph`")

                legend_html = '<div class="kg-legend">'
                for t, c in type_colors.items():
                    legend_html += (f'<div class="kg-legend-item">'
                                   f'<span class="kg-dot" style="background:{c}"></span>'
                                   f'{t}</div>')
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="section-card" style="text-align:center; padding:60px;">
                    <div style="font-size:3rem; margin-bottom:12px;">🕸</div>
                    <div style="font-weight:600;">Knowledge Graph Empty</div>
                    <div style="color:{COLORS['text_dim']}; margin-top:4px;">
                        Ingest memories with importance ≥ 0.4 to populate the graph
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-size:0.75rem; color:{COLORS['text_dim']}; margin-top:8px;">
                {kg_st.get('total_nodes', 0)} nodes ·
                {kg_st.get('total_edges', 0)} edges ·
                {kg_st.get('communities', '?')} communities
            </div>""", unsafe_allow_html=True)

        with sb_tabs[1]:
            contras = api_get("/contradictions")
            if contras and contras.get("count", 0):
                st.markdown(f"""<div style="font-size:0.85rem; color:{COLORS['red']};
                    font-weight:600; margin-bottom:12px;">
                    ⚠ {contras['count']} contradiction(s)</div>""",
                            unsafe_allow_html=True)
                for r in contras["reports"]:
                    st.markdown(f"""
                    <div class="inference-card contradiction">
                        <div style="font-weight:600;">{r.get('node_content', r.get('node_id', '?'))}</div>
                        <div style="color:{COLORS['text_dim']}; margin-top:4px; font-size:0.85rem;">
                            {r.get('analysis', r.get('description', ''))}
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No contradictions detected")

        with sb_tabs[2]:
            blinds = api_get("/blindspots")
            if blinds and blinds.get("count", 0):
                st.markdown(f"""<div style="font-size:0.85rem; color:{COLORS['yellow']};
                    font-weight:600; margin-bottom:12px;">
                    🔍 {blinds['count']} blindspot(s)</div>""",
                            unsafe_allow_html=True)
                for r in blinds["reports"]:
                    st.markdown(f"""
                    <div class="inference-card blindspot">
                        <div style="font-weight:600;">{r.get('node_content', r.get('node_id', '?'))}</div>
                        <div style="color:{COLORS['text_dim']}; margin-top:4px; font-size:0.85rem;">
                            {r.get('analysis', r.get('description', ''))}
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No blindspots detected")

        with sb_tabs[3]:
            threads = api_get("/threads")
            if threads and threads.get("count", 0):
                for t in threads["threads"]:
                    st.markdown(f"""
                    <div class="inference-card thread">
                        <div style="font-weight:600;">{t.get('topic', 'Unnamed')}</div>
                        <div style="color:{COLORS['text_dim']}; font-size:0.8rem;">
                            {t.get('size', '?')} nodes</div>
                        <div style="margin-top:6px; font-size:0.85rem;">
                            {t.get('summary', '')}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No threads discovered")

        with sb_tabs[4]:
            h1, h2 = st.columns([4, 1])
            with h1:
                st.markdown(f"""<div style="font-size:1.1rem; font-weight:700;">
                    Collision Insights</div>""", unsafe_allow_html=True)
            with h2:
                if st.button("⚡ Run", key="sb_rc", use_container_width=True):
                    api_post("/second-brain/collide", {"async": True})
                    st.toast("Collision started")

            report = api_get("/second-brain/report") or {}
            insights = report.get("recent_insights", [])
            if insights:
                for ins in insights[:10]:
                    nov = ins.get("novelty", "?")
                    strat = ins.get("strategy", "?")
                    ts = ins.get("timestamp", "")
                    nov_color = (COLORS["green"] if isinstance(nov, (int, float)) and nov >= 4
                                 else COLORS["yellow"] if isinstance(nov, (int, float)) and nov >= 3
                                 else COLORS["text_dim"])
                    st.markdown(f"""
                    <div class="insight-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="novelty" style="color:{nov_color};">{nov}</span>
                                <span style="color:{COLORS['text_dim']}; font-size:0.8rem; margin-left:8px;">
                                    {strat} · {fmt_time(ts)}</span>
                            </div>
                        </div>
                        <div style="margin-top:8px; font-size:0.9rem; line-height:1.5;">
                            {ins.get('text', ins.get('insight', ''))}
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.caption("No insights yet. Run a collision.")

        with sb_tabs[5]:
            report = api_get("/second-brain/report")
            if report:
                st.json(report)
            else:
                st.info("No report")


# ═══════════════════════════════════════════════════════════════════
# TAB 4-7: Context File Editors
# ═══════════════════════════════════════════════════════════════════

def render_editor(tab_w, key: str, fpath: Path, supplement_fn=None):
    with tab_w:
        exists = fpath.exists()
        mod = ""
        if exists:
            mod = datetime.fromtimestamp(fpath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline;
                    margin-bottom:16px;">
            <div>
                <span style="font-size:1.2rem; font-weight:700;">{fpath.name}</span>
                <span style="color:{COLORS['text_dim']}; font-size:0.8rem; margin-left:12px;">
                    {'Modified: ' + mod if mod else 'File not found'}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

        content = read_md(fpath) if exists else ""
        left, right = st.columns(2)
        with left:
            st.markdown(f'<div class="section-title">Editor</div>', unsafe_allow_html=True)
            edited = st.text_area("ed", value=content, height=400,
                                  key=f"ed_{key}", label_visibility="collapsed")
        with right:
            st.markdown(f'<div class="section-title">Preview</div>', unsafe_allow_html=True)
            st.markdown(edited)

        bc1, bc2, _ = st.columns([1, 1, 5])
        with bc1:
            if st.button("💾 Save", key=f"sv_{key}", type="primary",
                         use_container_width=True):
                try:
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(edited, encoding="utf-8")
                    st.toast(f"Saved {fpath.name}")
                except Exception as e:
                    st.error(str(e))
        with bc2:
            if st.button("↩ Revert", key=f"rv_{key}", use_container_width=True):
                st.rerun()

        if supplement_fn:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            supplement_fn()


def sup_memory():
    st.markdown(f'<div class="section-title">'
                f'<span class="dot" style="background:{COLORS["accent"]}"></span>'
                f'Daily Briefing</div>', unsafe_allow_html=True)
    b = api_get("/briefing") if online else None
    if b and "text" in b:
        st.markdown(f'<div class="briefing">{b["text"]}</div>', unsafe_allow_html=True)
    else:
        st.caption("Unavailable")


def sup_agents():
    st.markdown(f'<div class="section-title">'
                f'<span class="dot" style="background:{COLORS["green"]}"></span>'
                f'Session Context</div>', unsafe_allow_html=True)
    ctx = api_get("/session-context") if online else None
    if ctx:
        st.json(ctx)
    else:
        st.caption("Unavailable")


def sup_soul():
    st.markdown(f'<div class="section-title">'
                f'<span class="dot" style="background:{COLORS["purple"]}"></span>'
                f'Personality Profile</div>', unsafe_allow_html=True)
    pp = WORKSPACE / "memory" / "personality" / "PERSONALITY.yaml"
    if pp.exists():
        st.code(pp.read_text(encoding="utf-8"), language="yaml")
    else:
        st.caption("Not generated yet. Run `memory-cli consolidate`.")


render_editor(tabs[3], "memory", CONTEXT_FILES["Memory"], sup_memory)
render_editor(tabs[4], "agents", CONTEXT_FILES["Agents"], sup_agents)
render_editor(tabs[5], "user", CONTEXT_FILES["User"])
render_editor(tabs[6], "soul", CONTEXT_FILES["Soul"], sup_soul)
