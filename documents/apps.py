from django.apps import AppConfig
from .llm import create_knowledge_base

class DocumentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'documents'
    vector_stores = {}

    def ready(self):
        if not self.vector_stores:
            self.vector_stores = create_knowledge_base()
            print(f"Vector stores loaded into memory: {len(self.vector_stores)} documents...")
