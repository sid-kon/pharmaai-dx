import streamlit as st
from langchain_core.messages import HumanMessage
import base64

from agent.graph import pharmaai_graph, get_initial_state
from agent.state import DIMENSIONS

st.set_page_config(
    page_title="PharmaAI Dx",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── HIDE DEFAULT STREAMLIT TOP BAR ── */
    [data-testid="stToolbar"],
    header[data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    .stApp { background-color: #000000; }

    /* Push main content below the 80px fixed bar + small gap */
    .block-container {
        padding-top: 96px !important;
        margin-top: 0 !important;
    }

    /* Sidebar starts below the fixed bar, contents flush to top */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
        margin-top: 80px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }

    /* ── BOTTOM BAR ── */
    .stBottomBlockContainer,
    [data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer,
    [data-testid="stChatFloatingInputContainer"],
    section[data-testid="stBottom"],
    div.stBottom,
    footer.stFooter,
    .stApp > section > div:last-of-type,
    [class*="bottom"],
    [class*="Bottom"] {
        background-color: #161b27 !important;
        border-top: 1px solid #2a2f3e !important;
    }

    .dim-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 8px 12px; border-radius: 8px; margin-bottom: 6px;
        background: #1e2535; border: 1px solid #2a3245;
    }
    .dim-name { color: #c8d8e8; font-size: 0.82rem; font-weight: 500; }
    .badge { font-size: 0.75rem; font-weight: 600; padding: 2px 10px; border-radius: 12px; letter-spacing: 0.3px; }
    .badge-high    { background: #3d1515; color: #ff6b6b; border: 1px solid #ff4444; }
    .badge-medium  { background: #3d2e0a; color: #ffa726; border: 1px solid #ff8f00; }
    .badge-low     { background: #0d2e1a; color: #66bb6a; border: 1px solid #43a047; }
    .badge-pending { background: #1e2535; color: #546e7a; border: 1px solid #37474f; }

    .report-container {
        background: #161b27; border: 1px solid #2a2f3e;
        border-radius: 12px; padding: 28px; margin-top: 16px; color: #ffffff !important;
    }
    .report-container p, .report-container h1, .report-container h2,
    .report-container h3, .report-container li, .report-container td,
    .report-container th { color: #ffffff !important; }

    .success-banner {
        background: linear-gradient(135deg, #0d2e1a, #1a3d2b);
        border: 1px solid #43a047; border-radius: 10px;
        padding: 14px 20px; color: #66bb6a; font-weight: 600;
        font-size: 1rem; margin-bottom: 16px;
    }

    /* ── CHAT BUBBLES ── */
    [data-testid="stChatMessage"] {
        background: #161b27 !important;
        border: 1px solid #2a3245 !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] code { color: #ffffff !important; }

    /* ── CHAT INPUT BAR ── */
    [data-testid="stChatInput"] {
        background-color: #f5f5f5 !important;
        border-radius: 999px !important;
        border: 1.5px solid #e57373 !important;
        padding: 2px 12px !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        color: #333333 !important;
        font-size: 0.95rem !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        line-height: 1.5 !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
        resize: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #aaaaaa !important;
        font-style: normal !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #e57373 !important;
        border-radius: 50% !important;
        color: #ffffff !important;
        border: none !important;
        min-width: 32px !important;
        min-height: 32px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #e57373 !important;
        box-shadow: 0 0 0 2px rgba(229,115,115,0.15) !important;
    }

    /* ── DOWNLOAD BUTTON ── */
    [data-testid="stDownloadButton"] button {
        background-color: #161b27 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid #2a3245 !important;
        border-radius: 10px !important;
        width: 100% !important;
        padding: 14px 20px !important;
        font-size: 1rem !important;
        text-align: center !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background-color: #1e2535 !important;
        border-color: #4fc3f7 !important;
    }

    /* ── SIDEBAR BUTTON ── */
    [data-testid="stSidebar"] button {
        background-color: #1e2535 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid #2a3245 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] button:focus,
    [data-testid="stSidebar"] button:active {
        background-color: #1c6ea4 !important;
        color: #ffffff !important;
        border: 1px solid #4fc3f7 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* ── WHITE TEXT ── */
    .stMarkdown p, .stMarkdown li, .stMarkdown ul,
    .stMarkdown ol, .stMarkdown strong { color: #ffffff !important; }
    .stApp p, .stApp div, .stApp span, .stApp label { color: #ffffff; }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] strong { color: #ffffff !important; }

    .progress-label { color: #8ab4cc; font-size: 0.8rem; margin-bottom: 4px; }
    .sidebar-title {
        color: #4fc3f7; font-size: 1rem; font-weight: 700;
        letter-spacing: 0.5px; margin-bottom: 16px;
        padding-bottom: 10px; border-bottom: 1px solid #2a3245;
    }
</style>
""", unsafe_allow_html=True)

if "graph_state" not in st.session_state:
    st.session_state.graph_state = get_initial_state()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

SEVERITY_BADGE = {
    "High":         ("🔴", "badge-high",    "HIGH"),
    "Medium":       ("🟡", "badge-medium",  "MEDIUM"),
    "Low":          ("🟢", "badge-low",     "LOW"),
    "Not assessed": ("⚪", "badge-pending", "PENDING"),
}

def get_base64(img_path):
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

img_base64 = get_base64("agilisium_logo.jpeg")

logo_html = (
    f'<img src="data:image/jpeg;base64,{img_base64}" '
    f'style="height:36px; border-radius:6px; object-fit:contain;">'
    if img_base64 else "🧬"
)

# ── FIXED TOP BAR ────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 80px;
        background-color: #161b27;
        border-bottom: 1px solid #2a2f3e;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 999999;
        gap: 4px;
    ">
        <div style="display:flex; align-items:center; gap:12px;">
            {logo_html}
            <span style="font-size:1.5rem; font-weight:800; color:#ffffff; letter-spacing:-0.5px;">
                PharmaAI Dx
            </span>
        </div>
        <div style="font-size:0.78rem; color:#8ab4cc; letter-spacing:0.2px;">
            AI Pilot Implementation Diagnostic &nbsp;|&nbsp; Agilisium Consulting &nbsp;|&nbsp; Siddharth Kondubhatla
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    if img_base64:
        st.markdown(
            f'''
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;
                        padding-bottom:10px; border-bottom:1px solid #2a3245;">
                <img src="data:image/jpeg;base64,{img_base64}" width="30"
                     style="border-radius:6px;">
                <span style="font-size:18px; font-weight:bold; color:#4fc3f7;">PharmaAI Dx</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="sidebar-title">🧬 PharmaAI Dx</div>', unsafe_allow_html=True)

    st.markdown("**Diagnostic Progress**")

    scores = st.session_state.graph_state.get("dimension_scores", {})
    assessed_count = sum(1 for v in scores.values() if v != "Not assessed")

    st.markdown(
        f'<div class="progress-label">{assessed_count} of {len(DIMENSIONS)} dimensions assessed</div>',
        unsafe_allow_html=True,
    )
    st.progress(assessed_count / len(DIMENSIONS))
    st.markdown("<br>", unsafe_allow_html=True)

    for dim in DIMENSIONS:
        severity = scores.get(dim, "Not assessed")
        icon, badge_class, label = SEVERITY_BADGE.get(severity, SEVERITY_BADGE["Not assessed"])
        st.markdown(
            f'<div class="dim-row"><span class="dim-name">{dim}</span>'
            f'<span class="badge {badge_class}">{icon} {label}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start New Diagnostic", use_container_width=True):
        st.session_state.graph_state = get_initial_state()
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="color:#546e7a;font-size:0.75rem;">Powered by Agilisium Consulting'
        '<br>Built on LangGraph + Claude</div>',
        unsafe_allow_html=True,
    )

# ── COMPLETION BANNER + REPORT ────────────────────────────────────────
graph_state = st.session_state.graph_state

if graph_state.get("diagnosis_complete", False):
    st.markdown(
        '<div class="success-banner">✅ Full diagnostic complete — all 6 dimensions assessed</div>',
        unsafe_allow_html=True,
    )
    final_report = graph_state.get("final_report", "")
    if final_report:
        with st.expander("📋 View Full Diagnostic Report", expanded=True):
            st.markdown(
                f'<div class="report-container">{final_report}</div>',
                unsafe_allow_html=True,
            )
        st.download_button(
            label="⬇️ Download Diagnostic Report",
            data=final_report,
            file_name="pharmaai_dx_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ── CHAT HISTORY ──────────────────────────────────────────────────────
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown(
            "Welcome to **PharmaAI Dx**. I'm here to diagnose "
            "your pharma AI initiative.\n\n"
            "Please describe your initiative — what it does, what data it uses, "
            "how it's governed, where it sits in the implementation journey, "
            "and whether it touches regulated or clinical workflows. "
            "The more detail you share, the sharper the diagnosis."
        )

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Describe your pharma AI initiative...",
    disabled=graph_state.get("diagnosis_complete", False),
)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    full_description = "\n\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.chat_history
    ])

    current_state = st.session_state.graph_state
    current_state["messages"] = current_state.get("messages", []) + [
        HumanMessage(content=user_input)
    ]
    current_state["initiative_description"] = full_description

    with st.spinner("Analysing your initiative..."):
        try:
            result_state = pharmaai_graph.invoke(current_state)
        except Exception as e:
            result_state = current_state
            st.error(f"Graph invocation error: {e}")

    messages = result_state.get("messages", [])
    assistant_reply = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
            assistant_reply = msg.content
            break

    if assistant_reply:
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_reply,
        })

    st.session_state.graph_state = result_state
    st.rerun()