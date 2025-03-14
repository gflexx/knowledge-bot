from django.apps import AppConfig
from .llm import get_vector_store

class ChatsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chats'
    vector_store = None 

    def ready(self):
        if not self.vector_store: 
            # self.vector_store = get_vector_store()
            print("Vector store initialized...\n")
