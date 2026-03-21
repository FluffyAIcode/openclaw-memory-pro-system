import json
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="OpenClaw Memory Hub", layout="wide")
st.title("🧠 OpenClaw Memory Hub")
st.caption("Memora (RAG) + Chronos (CL) + MSA (Sparse Attention) | v1.2.0")

try:
    from memory_hub import hub
    st.success("Memory Hub connected")
except Exception as e:
    st.error(f"Failed to load Memory Hub: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Remember", "Recall", "Deep Recall", "Status"])

with tab1:
    st.subheader("Smart Remember")
    st.caption("Auto-routes to the right system(s) based on content length and importance")
    content = st.text_area("Content", height=200, placeholder="Paste text to remember...")
    col1, col2, col3 = st.columns(3)
    with col1:
        importance = st.slider("Importance", 0.0, 1.0, 0.7, 0.05)
    with col2:
        source = st.selectbox("Source", ["openclaw", "cli", "web", "manual"], index=0)
    with col3:
        title = st.text_input("Title (for MSA)", placeholder="Optional")

    word_count = len(content.split()) if content.strip() else 0
    targets = []
    if word_count > 0:
        targets.append("Memora")
    if word_count >= 100:
        targets.append("MSA")
    if importance >= 0.85:
        targets.append("Chronos")
    if targets:
        st.info(f"**{word_count} words** → will route to: {', '.join(targets)}")

    if st.button("💾 Remember", type="primary"):
        if content.strip():
            with st.spinner("Saving to memory systems..."):
                result = hub.remember(
                    content.strip(), source=source, importance=importance,
                    title=title if title else None)
            st.success(f"Remembered ({result['word_count']} words) via {', '.join(result['systems_used'])}")
            if "msa" in result:
                st.info(f"MSA doc: {result['msa']['doc_id']} ({result['msa']['chunks']} chunks)")
        else:
            st.warning("Please enter content")

with tab2:
    st.subheader("Merged Recall")
    st.caption("Searches across Memora (snippets) + MSA (documents)")
    query = st.text_input("Search query", key="recall_query")
    top_k = st.slider("Max results", 1, 20, 8, key="recall_k")
    if st.button("🔍 Recall") and query:
        with st.spinner("Searching..."):
            result = hub.recall(query, top_k=top_k)
        st.write(f"**Memora**: {len(result['memora'])} snippets | "
                 f"**MSA**: {len(result['msa'])} documents")
        for r in result["merged"]:
            sys_tag = r.get("system", "?")
            score = r.get("score", 0)
            meta = r.get("metadata", {})
            title_str = meta.get("title", meta.get("doc_id", ""))
            with st.expander(f"[{sys_tag}] score={score:.4f} {title_str}"):
                st.write(r.get("content", ""))

with tab3:
    st.subheader("Deep Recall (Multi-hop)")
    st.caption("Uses MSA Memory Interleave for cross-document reasoning")
    deep_query = st.text_input("Complex question", key="deep_query")
    max_rounds = st.slider("Max interleave rounds", 1, 5, 3, key="deep_rounds")
    if st.button("🔀 Deep Recall") and deep_query:
        with st.spinner("Running multi-hop reasoning..."):
            result = hub.deep_recall(deep_query, max_rounds=max_rounds)
        interleave = result.get("interleave")
        if interleave:
            st.metric("Rounds", interleave["rounds"])
            st.metric("Documents used", interleave["total_docs_used"])
            st.write("**Documents:**", ", ".join(interleave["doc_ids_used"]))
            st.text_area("Result", interleave["final_answer"][:2000], height=300)
        snippets = result.get("memora_context", [])
        if snippets:
            st.caption(f"+ {len(snippets)} Memora snippets for context")

with tab4:
    st.subheader("System Status")
    if st.button("🔄 Refresh"):
        st.rerun()
    try:
        status = hub.status()
        for name, info in status["systems"].items():
            if "error" in info:
                st.error(f"**{name}**: {info['error']}")
            else:
                with st.expander(f"**{name}**", expanded=True):
                    st.json(info)
    except Exception as e:
        st.error(f"Status check failed: {e}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Memora Digest"):
            with st.spinner("Running digest..."):
                hub.memora.auto_digest()
            st.success("Memora digest complete")
    with col2:
        if st.button("🔄 Chronos Consolidate"):
            with st.spinner("Consolidating..."):
                hub.chronos.consolidate()
            st.success("Chronos consolidation complete")

st.sidebar.info("OpenClaw Memory Hub v1.2.0\n4 systems integrated")
st.sidebar.write("**Current time:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
