import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_pdf_files(data_dir):
    """Load all PDF files from the given directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only the 'source' metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    """
    Split the given list of Document objects into smaller chunks
    using a RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


def download_hugging_face_embeddings():
    """
    Return HuggingFace embeddings served via the HF Inference API endpoint
    (no local model download).
    """
    os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings
