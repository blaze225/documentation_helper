from typing import Set
from backend.core import (
    run_llm_with_memory,
    initialize_llm,
    initialize_vectorstore,
)
import streamlit as st


def create_sources_string(source_urls: Set[str]) -> str:
    """Format source urls into a string.

    Args:
        source_urls (Set[str]): Set of source urls.

    Returns:
        str: source urls in a string.
    """
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if __name__ == "__main__":
    # Initialize vectorstore and LLM
    vectorstore = initialize_vectorstore()
    llm = initialize_llm(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

    # Set Title
    st.title("LangChain Documentation Helper")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user prompts
    if prompt := st.chat_input("Hi! Ask me anything about the Langchain Python API documentation."):
        # Add user prompt to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add a loader while LLM generates response
        with st.spinner("Generating response..."):
            # Call LLM to generate response
            response = run_llm_with_memory(
                query=prompt, llm=llm, vectorstore=vectorstore
            )
            sources = {doc.metadata["source"] for doc in response["context"]}
            formatted_response = f"{response['answer'].replace('Assistant:', '')}\n\n {create_sources_string(sources)}"
            print(formatted_response)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(formatted_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": formatted_response}
        )
