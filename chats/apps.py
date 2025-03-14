from django.apps import AppConfig
from django.apps import apps

class ChatsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chats'
    vector_store = None 

    def ready(self):
        if not self.vector_store: 
            documents_config = apps.get_app_config('documents')
            # if documents_config.vector_store is None:
            #     raise Exception("Vector store not initialized or Empty. Check documents/apps.py")

            # return documents_config.vector_store.as_retriever(
            #     search_type="mmr",
            #     search_kwargs={
            #         "k": 3,
            #         "filter": {"priority": True}
            #     }
            # )
