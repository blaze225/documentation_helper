import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceEndpoint

load_dotenv()


def ingest_docs() -> None:
    """Read all documents from langchain-docs, split into chunks"""
    loader = ReadTheDocsLoader(
        path="langchain-docs/api.python.langchain.com/en/latest/"
    )
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents.")
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks.")

    # Update source in metadata to a url
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"{len(documents)} document chunks ready to be ingested into Pinecone.")
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()
    # Ingest into the vector store
    PineconeVectorStore.from_documents(
        documents=documents, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )


if __name__ == "__main__":
    ingest_docs()
