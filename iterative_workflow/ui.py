import streamlit as st
from agent_backend import app, MAX_ATTEMPTS

st.set_page_config(page_title="LinkedIn Post Generator", page_icon="✍️", layout="centered")

st.title("✍️ LinkedIn Post Generator")
st.caption("Writer agent (Mistral) + Reviewer agent (Groq/Llama) with web search & auto-retry loop.")

with st.form("topic_form"):
    topic = st.text_input("Enter your topic", placeholder="e.g. why AI agents are the next big shift in software engineering")
    submitted = st.form_submit_button("Generate Post")

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    status_box = st.status("Starting agent pipeline...", expanded=True)

    try:
        with status_box:
            st.write("🔎 Writer drafting (may call web search)...")
            result = app.invoke({
                "topic": topic,
                "messages": [],
                "draft": "",
                "review_feedback": "",
                "isApproved": False,
                "attempt": 0,
            })
            st.write(f"📝 Draft generated after {result['attempt']} attempt(s).")
            st.write("🧐 Reviewer evaluated the final draft.")
        status_box.update(label="Done!", state="complete", expanded=False)

    except Exception as e:
        status_box.update(label="Failed", state="error")
        st.error(f"Something went wrong: {e}")
        st.stop()

    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Final Post")
    with col2:
        if result["isApproved"]:
            st.success("Approved ✅")
        else:
            st.warning(f"Not approved (hit {MAX_ATTEMPTS}-attempt cap) ⚠️")

    st.text_area("Post content", value=result["draft"], height=280)

    st.download_button(
        "⬇️ Download as .txt",
        data=result["draft"],
        file_name="linkedin_post.txt",
        mime="text/plain",
    )

    with st.expander("Reviewer feedback"):
        st.write(result["review_feedback"] or "No feedback recorded.")

    with st.expander(f"Attempts used: {result['attempt']} / {MAX_ATTEMPTS}"):
        st.json({
            "attempt": result["attempt"],
            "isApproved": result["isApproved"],
        })