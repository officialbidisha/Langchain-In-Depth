from typing import Set

import streamlit as st
from streamlit_chat import message

from backend.core import run_llm

st.set_page_config(page_title="LangChain Documentation Helper", page_icon="🦜")
st.header("LangChain Documentation Helper Bot 🦜")

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []


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
    with st.spinner("Generating response..."):
        response = run_llm(query=prompt)
        sources = {doc["source"] for doc in response["context"]}
        formatted_response = f"{response['answer']}{create_sources_string(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for i, (user_query, generated_response) in enumerate(
        zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answers_history"],
        )
    ):
        message(user_query, is_user=True, key=f"user_{i}")
        message(generated_response, key=f"bot_{i}")
