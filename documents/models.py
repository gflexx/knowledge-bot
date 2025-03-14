from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from django.apps import apps
from django.db import models


class Document(models.Model):
    title = models.CharField(max_length=252)
    file = models.FileField(upload_to="documents/")
    priority = models.BooleanField(default=False)
    creation_time = models.DateTimeField(auto_now_add=True)
    last_updated_time = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-creation_time',]

    def __str__(self):
        return self.title
    
    def extract_text(self):
        """
        extract text from the document based on file type
        """
        return
    
    def add_to_vector_store(self):
        """
        process the document and add it to the vector store
        """
        return
    
