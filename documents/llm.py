from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from django.apps import apps
import os

document_chroma = "./document_chroma"
hagging_face_embeddings = "BAAI/bge-small-en-v1.5"
recreate_vector_store = False


def get_vector_store():
    """
    load the vector store if it exists
    """
    if os.path.exists(document_chroma) and os.listdir(document_chroma):
        print("Loading existing vector store...")
        embeddings = HuggingFaceEmbeddings(model_name=hagging_face_embeddings)
        return Chroma(
            persist_directory=document_chroma, 
            embedding_function=embeddings
        )
    
    else:
        print("No existing vector store found...")
        return None 
    

def create_knowledge_base():
    """
    create a new vector store from all existing documents
    """
    print("Initializing vector store from existing documents...")

    documents = Document.objects.all()
    all_texts = []

    for doc in documents:
        text = doc.extract_text()
        if text:
            all_texts.append(
                {
                    "page_content": text, 
                    "metadata": {
                        "title": doc.title,
                        "priority": doc.priority
                    }
                }
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=250
    )
    split_docs = text_splitter.split_documents(all_texts)

    embeddings = HuggingFaceEmbeddings(model_name=hagging_face_embeddings)
    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=document_chroma
    )

    return vectorstore


def add_document_to_vector_store(document):
    """
    add a single document to the vector store
    """
    documents_config = apps.get_app_config('documents')

    if documents_config.vector_store:
        text = document.extract_text()
        if text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=250
            )
            split_docs = text_splitter.split_documents(
                [
                    {
                        "page_content": text, 
                        "metadata": {
                            "title": document.title,
                            "priority": document.priority
                        }
                    }
                ]
            )

            documents_config.vector_store.add_documents(documents=split_docs)
            documents_config.vector_store.persist()
            print(f"{document.title} added to vector store.")