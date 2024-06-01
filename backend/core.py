import os
from typing import Any
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

warnings.filterwarnings("ignore")

store = {}


def run_llm(
    query: str, llm: HuggingFaceEndpoint, vectorstore: PineconeVectorStore
) -> Any:
    # Prompt sent to the LLM after retreival from vector store
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # Chain to stuff documents into a prompt and then pass that to an LLM
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )
    # Chain to retrieve information from the vector store and run the combine_docs_chain
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    print("**** Running retrieval chain...")
    # Invoke the retrieval chain
    result = retrieval_chain.invoke(input={"input": query})
    return result


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def initialize_vectorstore() -> PineconeVectorStore:
    print("Initializing vector store...")
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()
    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    return vectorstore


def initialize_llm(model: str) -> HuggingFaceEndpoint:
    print("Initializing LLM...")
    llm = HuggingFaceEndpoint(
        repo_id=model,
        temperature=0.1,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )
    return llm


def run_llm_with_memory(
    query: str, llm: HuggingFaceEndpoint, vectorstore: PineconeVectorStore
) -> Any:

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    result = conversational_rag_chain.invoke(
        input={"input": query}, config={"configurable": {"session_id": "llm1"}}
    )
    return result


if __name__ == "__main__":

    vectorstore = initialize_vectorstore()
    llm = initialize_llm(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    llm_response = run_llm_with_memory(
        query="What are chains in Langchain?", llm=llm, vectorstore=vectorstore
    )
    print(f"Answer:\n{llm_response['answer']}")
    print(f"Sources: {[doc.metadata['source'] for doc in llm_response['context']]}")

    llm_response = run_llm_with_memory(
        query="Why are they used?", llm=llm, vectorstore=vectorstore
    )
    print(f"Answer:\n{llm_response['answer']}")
    print(f"Sources: {[doc.metadata['source'] for doc in llm_response['context']]}")
