from rest_framework.generics import GenericAPIView
from django.http import JsonResponse
from django.http import HttpRequest

from .serializers import *


class CheckConversation(GenericAPIView):
    """
    gets or creates conversation cookie
    """
    serializer_class = ConversationSerializer
    
    def get(self, request: HttpRequest, *args, **kwargs):
        pass
