import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Memora 记忆系统", layout="wide")
st.title("🧠 Memora 记忆管理系统")
st.caption("OpenClaw 增强型记忆层 | v1.0.0")

try:
    from memora.bridge import bridge
    from memora.config import load_config

    config = load_config()
    st.success("✅ Memora 桥接器已连接")
except Exception as e:
    st.error(f"加载失败: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["添加记忆", "搜索记忆", "系统状态"])

with tab1:
    st.subheader("添加新记忆")
    content = st.text_area("记忆内容", height=150, placeholder="今天学到了...")
    importance = st.slider("重要性", 0.0, 1.0, 0.7, 0.05)
    source = st.selectbox("来源", ["cli", "openclaw", "web", "manual"], index=1)

    if st.button("💾 保存记忆", type="primary"):
        if content.strip():
            try:
                with st.spinner("正在双写保存..."):
                    bridge.save_to_both(content.strip(), source=source, importance=importance)
                st.success("记忆已成功保存！")
                st.balloons()
            except Exception as e:
                st.error(f"保存失败: {e}")
        else:
            st.warning("请输入记忆内容")

with tab2:
    st.subheader("搜索记忆")
    query = st.text_input("搜索关键词")
    if st.button("🔍 搜索") and query:
        try:
            with st.spinner("搜索中..."):
                results = bridge.search_across(query)
            if results:
                st.write(f"找到 **{len(results)}** 条相关记忆")
                for r in results:
                    score = r.get("score", "")
                    label = f"[{score}] " if score else ""
                    st.info(f"{label}{r.get('content', str(r))}")
            else:
                st.info("未找到相关记忆")
        except Exception as e:
            st.error(f"搜索失败: {e}")

with tab3:
    st.subheader("系统状态")
    st.write("**当前时间**:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("**记忆存储路径**:", str(config.base_dir.resolve()))

    from memora.vectorstore import vector_store
    st.write("**向量条目数**:", vector_store.count())

    if st.button("⚡ 执行记忆提炼"):
        try:
            with st.spinner("正在提炼..."):
                bridge.auto_digest()
            st.success("记忆提炼完成！")
        except Exception as e:
            st.error(f"提炼失败: {e}")

st.sidebar.info("Memora + OpenClaw 集成\n双写机制已启用")
