# documentation_helper

An StreamLit based app for Question Answering on Langchain documentation.

## Application

https://huggingface.co/spaces/saadkh225/langchain-documentation-helper

## Tools used

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.1-3776AB.svg?style=flat&logo=LangChain&logoColor=white)](https://python.langchain.com/v0.2/docs/introduction/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Approach

LLMs with Conversational Retrieval Augmented Generation (RAG).

## Data

Langchain Documentation: https://api.python.langchain.com/en/latest/langchain_api_reference.html

## Steps

1. Download LangChain documentation

    ``` shell
    wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/latest/langchain_api_reference.html
    ```
    This will download web pages into `langchain-docs/`

2. Create embeddings and ingest into vector store

    Embeddings used: `HuggingFaceEmbeddings`

    VectorDB used: `Pinecone`

3. Create a conversational RAG chain

    LLM used: `Mixtral-8x7B-Instruct-v0.1`

    https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/

4. Integrate Steamlit based frontend

    https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

