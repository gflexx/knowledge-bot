from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
import google.generativeai as genai

from django.apps import apps
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

document_chroma = "./document_chroma"
hagging_face_embeddings = "BAAI/bge-small-en-v1.5"
recreate_vector_store = False

genai.configure(api_key=GOOGLE_API_KEY)


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
    DocumentModel = apps.get_model('documents', 'Document')
    documents = DocumentModel.objects.all()
    all_texts = []

    for doc in documents:
        extracted_pages = doc.extract_text() 

        for page in extracted_pages:
            all_texts.append(
                Document( 
                    page_content=page["text"],
                    metadata={
                        "document_id": str(doc.id),
                        "title": doc.title,
                        "priority": doc.priority,
                        "page": page["page"] 
                    }
                )
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
        extracted_pages = document.extract_text()
        
        for page in extracted_pages:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=250
            )
            split_docs = text_splitter.split_documents(
                [
                    Document( 
                        page_content=page["text"],
                        metadata={
                            "document_id": str(document.id),
                            "title": document.title,
                            "priority": document.priority,
                            "page": page["page"] 
                        }
                    )
                ]
            )

            documents_config.vector_store.add_documents(documents=split_docs)
            print(f"{document.title} added to vector store.")


# prompt template
template = """
    You are a Company Knowledge Base Assistant.
    Your role is to provide accurate and concise answers based on the provided company knowledge base.
    Use only the context given to answer the question. Avoid assumptions and repetition.
    If the context does not contain enough information, respond with "I don't have enough information to answer."
    {context}
    Question: {question}
    Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)


def get_retriever():
    """
    get the retriever from the app's vector store.
    """
    documents_config = apps.get_app_config('documents')
    if documents_config.vector_store is None:
        raise Exception("Vector store not initialized or Empty. Check documents/apps.py")
    
    vector_store = documents_config.vector_store
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "lambda_mult": 0.5,
            # "filter": {"priority": True}
        },
        
    )


async def gemini_stream_call(prompt):
    """
    streams response from Google Gemini API.
    """
    model = genai.GenerativeModel()

    print(f"GEMINI {model.__class__.__name__}")

    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        yield chunk.text


async def stream_answer(inputs):
    """
    retrieve context, format prompt, and stream response from Google Gemini.
    """
    question = inputs["question"]
    retriever = get_retriever()
    context_docs = retriever.invoke(question)

    print(f"Question: {question}")

    formatted_context = "\n".join([doc.page_content for doc in context_docs])

    references = []
    seen_references = set()

    for i, doc in enumerate(context_docs):
        source = doc.metadata.get('title', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        reference = f"Source: {source}, Page: {page}"

        if reference not in seen_references:
            seen_references.add(reference)
            references.append(f"[{len(references) + 1}] {reference}")

    references_text = "\n".join(references)

    formatted_prompt = prompt.format(context=formatted_context, question=question)
    
    async for value in gemini_stream_call(formatted_prompt):
        yield value

    print(references_text)

    if references_text:
        yield f"\nReferences:\n{references_text}"