import os

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import download_hugging_face_embeddings, filter_to_minimal_docs, load_pdf_files, text_split
from src.prompt import system_prompt
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

INDEX_NAME = "medical-chatbot"


def build_docsearch(embeddings):
    """
    Connect to the Pinecone index, creating and populating it from the
    PDFs in data/ if it doesn't exist yet, otherwise reuse it as-is.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if pc.has_index(INDEX_NAME):
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
        )

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    extracted_data = load_pdf_files("data")
    minimal_docs = filter_to_minimal_docs(extracted_data)
    texts_chunk = text_split(minimal_docs)

    return PineconeVectorStore.from_documents(
        documents=texts_chunk,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )


def build_rag_chain(docsearch):
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    chat_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def main():
    embeddings = download_hugging_face_embeddings()
    docsearch = build_docsearch(embeddings)
    rag_chain = build_rag_chain(docsearch)

    response = rag_chain.invoke({"input": "What is Acne?"})
    print(response["answer"])


if __name__ == "__main__":
    main()
