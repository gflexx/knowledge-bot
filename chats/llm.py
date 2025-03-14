from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import torch
import glob
import os
from django.apps import apps
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
scrape_dir = "./scraped_content"
hagging_face_embeddings = "BAAI/bge-small-en-v1.5"
recreate_vector_store = False

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

genai.configure(api_key=GOOGLE_API_KEY)


def load_scraped_documents():
    """
    load all scraped documents into objects
    """
    documents = []
    text_files = glob.glob(os.path.join(scrape_dir, "*.txt"))

    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                filename = os.path.basename(file_path)
                page_name = os.path.splitext(filename)[0]
                priority = page_name in [
                    "home", "portfolio", "services_mobile_app_developers_in_kenya_1",
                    "onboarding"
                ]
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "page": page_name,
                        "priority": priority
                    }
                )
                documents.append(doc)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(documents)} documents from scraped content...")
    return documents


def create_knowledge_base():
    """
    create a knowledge base from the scraped documents
    """
    print("Creating knowledge base...")

    documents = load_scraped_documents()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks...")

    embeddings = HuggingFaceEmbeddings(model_name=hagging_face_embeddings)
    unique_docs = list({doc.page_content: doc for doc in split_docs}.values())

    vectorstore = Chroma.from_documents(
        documents=unique_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore


def get_vector_store():
    """
    load or create vector store.
    """
    if os.path.exists("./chroma_db") and os.listdir("./chroma_db") and not recreate_vector_store:
        print("Loading existing vector store...")
        embeddings = HuggingFaceEmbeddings(model_name=hagging_face_embeddings)
        return Chroma(
            persist_directory="./chroma_db", 
            embedding_function=embeddings
        )

    else:
        return create_knowledge_base()

# prompt template
template = """
    You are Glitex Solutions' AI assistant.
    Answer the question using the context provided. Avoid repetition.
    If the context does not contain enough information, say "I don't have enough information to answer."
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
    chats_config = apps.get_app_config('chats')
    if chats_config.vector_store is None:
        raise Exception("Vector store not initialized.  Check chats/apps.py")
    return chats_config.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "filter": {"priority": True}
        }
    )


def format_prompt(inputs):
    """
    retrieve context from LangChain retriever and format the prompt.
    """
    question = inputs["question"]
    retriever = get_retriever()
    context_docs = retriever.invoke(question)

    formatted_context = "\n".join([doc.page_content for doc in context_docs])

    sources = [doc.metadata.get("source", "Unknown source") for doc in context_docs]
    
    return {
        "prompt": prompt.format(context=formatted_context, question=question),
        "sources": sources
    }


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

    print(question)

    formatted_context = "\n".join([doc.page_content for doc in context_docs])
    references = []

    for i, doc in enumerate(context_docs):
        references.append(f"[{i+1}] Source: {doc.metadata.get('source', 'Unknown')}")

    references_text = "\n".join(references)

    formatted_prompt = prompt.format(context=formatted_context, question=question)
    
    async for value in gemini_stream_call(formatted_prompt):
        yield value

    yield f"\nReferences:\n{references_text}"