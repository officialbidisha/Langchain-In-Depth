import time
from typing import Set

import streamlit as st
from streamlit_chat import message

from backend.core import run_llm

# Public demo guardrails: each query hits paid OpenAI + Pinecone APIs, so cap
# usage per browser session rather than leaving it unbounded.
MAX_QUERIES_PER_SESSION = 20
MIN_SECONDS_BETWEEN_QUERIES = 3
MAX_QUERY_LENGTH = 500

st.set_page_config(page_title="LangChain Documentation Helper", page_icon="🦜")
st.header("LangChain Documentation Helper Bot 🦜")

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["query_count"] = 0
    st.session_state["last_query_time"] = 0.0

remaining = MAX_QUERIES_PER_SESSION - st.session_state["query_count"]
st.caption(f"{remaining} question(s) left in this session")


def create_sources_string(sources: Set[str]) -> str:
    if not sources:
        return ""
    sources_list = sorted(sources)
    sources_string = "\n\n**Sources:**\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


prompt = st.chat_input("Ask something about the LangChain docs")

if prompt:
    now = time.time()
    seconds_since_last = now - st.session_state["last_query_time"]

    if len(prompt) > MAX_QUERY_LENGTH:
        st.warning(f"Please keep questions under {MAX_QUERY_LENGTH} characters.")
    elif st.session_state["query_count"] >= MAX_QUERIES_PER_SESSION:
        st.warning(
            f"You've reached the demo limit of {MAX_QUERIES_PER_SESSION} questions "
            "for this session. Refresh the page to start a new session."
        )
    elif seconds_since_last < MIN_SECONDS_BETWEEN_QUERIES:
        st.warning(
            f"Please wait {MIN_SECONDS_BETWEEN_QUERIES - seconds_since_last:.1f}s "
            "before asking another question."
        )
    else:
        with st.spinner("Generating response..."):
            response = run_llm(query=prompt)
            sources = {doc["source"] for doc in response["context"]}
            formatted_response = f"{response['answer']}{create_sources_string(sources)}"

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["query_count"] += 1
            st.session_state["last_query_time"] = now

if st.session_state["chat_answers_history"]:
    for i, (user_query, generated_response) in enumerate(
        zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answers_history"],
        )
    ):
        message(user_query, is_user=True, key=f"user_{i}")
        message(generated_response, key=f"bot_{i}")
