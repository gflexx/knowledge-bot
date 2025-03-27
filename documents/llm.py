from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
import torch
from huggingface_hub import login
from transformers import pipeline

from asgiref.sync import sync_to_async
import asyncio

from django.apps import apps
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

document_chroma = "./document_chroma"
hagging_face_embeddings = "BAAI/bge-base-en-v1.5"
recreate_vector_store = False

print("Loading Models...")

# print("Logging in to Hugging Face...")
# login(token=HF_API_KEY)

# init models
hf_embeddings = HuggingFaceEmbeddings(
    model_name=hagging_face_embeddings,
    model_kwargs={'device': 'cpu'},
)

transformers_model = SentenceTransformer(
    hagging_face_embeddings, 
    device="cpu",
)

ner_pipeline = pipeline(
    "ner", 
    model="dslim/bert-base-NER", 
    aggregation_strategy="simple"
)

genai.configure(api_key=GOOGLE_API_KEY)


def get_document_vector_store_dir(document_id):
    return os.path.join(document_chroma, f"doc_{document_id}")


def get_vector_store(document_id):
    """
    load a document's vector store from disk into memory.
    """
    doc_vector_store_dir = get_document_vector_store_dir(document_id)
    if os.path.exists(doc_vector_store_dir) and os.listdir(doc_vector_store_dir):
        print(f"Loading vector store for document {document_id}...")

        return Chroma(
            persist_directory=doc_vector_store_dir, 
            embedding_function=hf_embeddings
        )
    
    print(f"No vector store found for document {document_id}, initializing empty store...")
    return None
    

def create_knowledge_base():
    print("Initializing individual vector stores from existing documents...")
    DocumentModel = apps.get_model('documents', 'Document')
    documents = DocumentModel.objects.all()

    vector_stores = {}
    for doc in documents:
        store = get_vector_store(doc.id)
        if store:
            vector_stores[doc.id] = store
    
    print(f"Loaded {len(vector_stores)} vector stores into memory.")
    return vector_stores


def add_document_to_vector_store(document):
    """
    create a separate vector store for a single document.
    """
    documents_config = apps.get_app_config("documents")
    vector_stores = documents_config.vector_stores 
    doc_vector_store_dir = get_document_vector_store_dir(document.id)
    os.makedirs(doc_vector_store_dir, exist_ok=True)

    extracted_pages = document.extract_text()
    all_texts = []

    for page in extracted_pages:
        all_texts.append(
            Document(
                page_content=page["text"],
                metadata={
                    "document_id": str(document.id),
                    "title": document.title,
                    "priority": document.priority,
                    "page": page["page"]
                }
            )
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""], 
        keep_separator=False
    )
    split_docs = text_splitter.split_documents(all_texts)

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=hf_embeddings,
        persist_directory=doc_vector_store_dir
    )

    vector_stores[document.id] = vector_store

    print(f"Vector store created for document {document.id}: {document.title}")


# prompt template
template = """
    You are a Company Knowledge Base Assistant.
    Your role is to provide accurate and concise answers based only on the provided company knowledge base.

    Answer the user's question truthfully and clearly using the information available in the context below.
    If the information is incomplete or missing, do the following:
    - If you can infer an answer based on the company's services and purpose, provide a helpful general response.
    - If no reasonable information is found in the context, reply: "I don't have enough information to answer."

    Avoid making assumptions, repeating the question, or providing external information.
    Use clear and simple language.

    Context: {context}
    Question: {question}
    Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)


async def detect_source_from_question(question):
    DocumentModel = apps.get_model('documents', 'Document')
    sources = await sync_to_async(list)(DocumentModel.objects.values_list('title', flat=True))

    ner_results = ner_pipeline(question)
    entities = []
    current_entity = []

    # Merge consecutive entities
    for result in ner_results:
        if result["entity_group"] in ["ORG", "MISC", "PRODUCT", "WORK_OF_ART"]:
            word = result["word"].replace("##", "")
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(" ".join(current_entity).strip())
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity).strip())

    print(f"NER Entities: {entities}")

    # Fuzzy Match each entity to sources
    for entity in entities:
        match, score, _ = process.extractOne(
            entity,
            sources,
            scorer=fuzz.partial_ratio  # You can try ratio, token_set_ratio, etc.
        )
        print(f"Trying match: '{entity}' -> '{match}' (Score: {score})")
        if score >= 50:  # You can tweak this threshold
            return match

    return None



def get_retriever(document_id, vector_stores):
    vector_store = vector_stores.get(document_id)
    if vector_store is None:
        raise Exception(f"Vector store not found for document {document_id}")
    
    
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5, "k": 3
        }
    )


async def get_hybrid_retriever(question, document_id, vector_stores):
    """
    Combines semantic and keyword retrievers, deduplicates, and re-ranks results.
    """
    vector_store = vector_stores.get(document_id)
    if vector_store is None:
        raise Exception(f"Vector store not found for document {document_id}")

    # Semantic search retriever
    semantic_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3}
    )

    keyword_retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    semantic_results, keyword_results = await asyncio.gather(
        sync_to_async(semantic_retriever.invoke)(question),
        sync_to_async(keyword_retriever.invoke)(question)
    )

    combined = {doc.page_content: doc for doc in semantic_results + keyword_results}
    combined_results = list(combined.values())

    reranked_docs = await re_rank_results(question, combined_results)

    return reranked_docs


async def re_rank_results(question, documents):
    """
    Re-rank documents based on cosine similarity to the question.
    """
    if not documents:
        return []

    question_embedding = await asyncio.to_thread(
        transformers_model.encode,
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scored_docs = []
    for doc in documents:
        doc_embedding = await asyncio.to_thread(
            transformers_model.encode,
            doc.page_content,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        score = util.pytorch_cos_sim(question_embedding, doc_embedding).item()
        scored_docs.append((doc, score))

    reranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    for doc, score in reranked:
        print(f"Score: {score:.2f} | Source: {doc.metadata.get('title', 'Unknown')}")

    return [doc for doc, score in reranked]


async def gemini_stream_call(prompt):
    """
    streams response from Google Gemini API.
    """
    model = genai.GenerativeModel()

    print(f"GEMINI {model.__class__.__name__}")

    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        yield chunk.text


def extract_and_join_page_contents(documents, separator=" "):
    page_contents = []
    for document in documents:
        if hasattr(document, 'page_content') and document.page_content:
            cleaned_text = document.page_content.replace("\n", " ")  # Remove all newlines
            page_contents.append(cleaned_text)

    joined_text = separator.join(page_contents)
    return joined_text


async def compute_similarity(doc_id, context_docs, question_embedding):
    """
    filter out unwanted ctx docs by comparing transformer similarity
    """
    if not context_docs:
        return None
    
    doc_text = extract_and_join_page_contents(context_docs)

    #Use run_in_executor to run CPU-intensive encoding in a thread pool
    text_embedding = await asyncio.to_thread(
        transformers_model.encode, 
        doc_text, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    similarity_score = util.pytorch_cos_sim(question_embedding, text_embedding).item()

    print(f"Similarity score {similarity_score} -- {doc_id}")
    
    if similarity_score >= 0.3:
        return (doc_id, context_docs)
    return None


async def stream_answer(inputs):
    """
    retrieve context, format prompt, and stream response from Google Gemini.
    """
    documents_config = apps.get_app_config("documents")
    vector_stores = documents_config.vector_stores 

    question = inputs["question"]

    source_title = await detect_source_from_question(question)
    print(f"Detected Source: {source_title}")

    DocumentModel = apps.get_model('documents', 'Document')

    if source_title:
        all_documents = await sync_to_async(list)(DocumentModel.objects.filter(title=source_title))
    else:
        all_documents = await sync_to_async(list)(DocumentModel.objects.all())

    relevant_docs = []
    async def fetch_context(doc):
        context_docs = await get_hybrid_retriever(question, doc.id, vector_stores)
        return doc.id, context_docs

    tasks = [fetch_context(doc) for doc in all_documents]
    results = await asyncio.gather(*tasks)

    # load question to transformers
    question_embedding = transformers_model.encode(question, convert_to_tensor=True, normalize_embeddings=True)

    similarity_tasks = [
        compute_similarity(
            doc_id, context_docs, question_embedding
        ) 
            for doc_id, context_docs in results
        ]

    # run all similarity computations concurrently
    similarity_results = await asyncio.gather(*similarity_tasks)

    # filter out None results
    relevant_docs = [result for result in similarity_results if result is not None]

    if not relevant_docs:
        yield "No relevant documents found."
        return

    all_contexts = []
    unique_references = {} 

    for doc_id, context_docs in relevant_docs:
        for doc in context_docs:
            all_contexts.append(doc.page_content)
            # print(relevant_docs)
            
            source = doc.metadata.get('title', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            key = f"{source}_{page}"
            
            # Only add each unique source/page combination once
            if key not in unique_references:
                unique_references[key] = {
                    "source": source,
                    "page": int(page) if str(page).isdigit() else 0, 
                    "doc_id": doc_id
                }

    if not all_contexts:
        yield "No relevant content found in selected documents."
        return

    formatted_context = " ".join(" ".join(all_contexts).split())
    all_references = []
    sorted_references = sorted(
        unique_references.values(),
        key=lambda x: (x["source"], int(x["page"]))  
    )
    for i, ref_text in enumerate(sorted_references):
        ref_text = f"Source: {ref_text['source']}, Page: {ref_text['page']} <hidden={ref_text['doc_id']}>"
        all_references.append(f"[{i+1}] {ref_text}")

    references_text = "\n".join(all_references)

    formatted_prompt = prompt.format(context=formatted_context, question=question)

    print(formatted_context)

    # async for value in gemini_stream_call(formatted_prompt):
    #     yield value

    if references_text:
        yield f"\nReferences:\n{references_text}"