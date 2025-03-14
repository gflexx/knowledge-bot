from django.apps import AppConfig
from .llm import get_vector_store

class DocumentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'documents'
    vector_store = None

    def ready(self):
        if not self.vector_store:
            self.vector_store = get_vector_store()
            if self.vector_store:
                print(f"{self.name} vector store initialized...")
