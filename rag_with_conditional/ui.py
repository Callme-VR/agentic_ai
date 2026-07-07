# this file for building the ui using ai agent made by me based the project requirements and the problem statement given in the agent.py file. The ui will be built using gradio library and will have 3 conditional paths based on the user input. The user will choose one of the 3 options and then a chatbot will be activated. The chatbot will answer the questions based on the selected option using RAG retriever for academic and fee related questions and LLM for general questions. The responses from all 3 paths will converge to a single node.


import streamlit as st
from agent import app
# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#F6F8FC;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    font-size:40px;
    font-weight:700;
    color:#1E3A8A;
}

.subtitle{
    font-size:18px;
    color:#555;
    margin-bottom:20px;
}

.chat-user{

    background:#2563EB;
    color:white;
    padding:15px;
    border-radius:15px;
    margin-bottom:10px;
}

.chat-ai{

    background:#F8FAFC;
    color:#1E293B;
    padding:15px;
    border-radius:15px;
    border:1px solid #E2E8F0;
    margin-bottom:20px;

}

.stButton>button{

    width:100%;
    border-radius:10px;
    height:45px;
    background:#2563EB;
    color:white;
    border:none;
    font-weight:bold;

}

.stButton>button:hover{

    background:#1D4ED8;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "programme" not in st.session_state:
    st.session_state.programme = "BCA"

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=120
    )

    st.title("Student Details")

    programme = st.selectbox(
        "Select Programme",
        [
            "BCA",
            "B.Com",
            "BBA"
        ]
    )

    st.session_state.programme = programme

    st.divider()

    st.write("### About")

    st.write(
        """
College AI Assistant

Supports

✅ Academic Questions

✅ Fee Queries

✅ General Questions

Powered by

LangGraph

Groq

FAISS

Sentence Transformers
        """
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ==========================================================
# HEADER
# ==========================================================

st.markdown('<div class="title">🎓 College RAG Assistant</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Ask questions about academics, fees, attendance, exams and more.</div>',
    unsafe_allow_html=True
)

# ==========================================================
# DISPLAY CHAT
# ==========================================================

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f"""
<div class="chat-user">
<b>You</b><br>
{msg["content"]}
</div>
""",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
<div class="chat-ai">
<b>College Assistant</b><br>
{msg["content"]}
</div>
""",
            unsafe_allow_html=True,
        )

# ==========================================================
# CHAT INPUT
# ==========================================================

user_input = st.chat_input("Ask your question...")

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.spinner("Thinking..."):

        result = app.invoke(
            {
                "Programme": st.session_state.programme,
                "messages": [
                    ("human", user_input)
                ]
            }
        )

        answer = result["messages"][-1].content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()