from rest_framework.generics import GenericAPIView, ListAPIView
from django.http import JsonResponse
from django.http import HttpRequest

from rest_framework.exceptions import NotFound
from rest_framework import status


from .serializers import *


class CheckConversation(GenericAPIView):
    """
    gets or creates conversation cookie
    """
    serializer_class = ConversationSerializer

    def get(self, request: HttpRequest, *args, **kwargs):
        conversation, is_new = Conversation.objects.check_conversation(request)

        data = self.serializer_class(conversation).data

        response = JsonResponse(data=data)
        
        if is_new:
            response.set_cookie(
                'conversation_id', 
                conversation.id,
                samesite='None',
            )
        return response
    

class StartConversation(GenericAPIView):
    """
    starts a new conversation
    """
    serializer_class = ConversationSerializer

    def post(self, request):
        conversation = Conversation.objects.create()
        data = self.serializer_class(conversation).data

        response = JsonResponse(data=data)
        response.set_cookie(
            'conversation_id', 
            conversation.id,
            samesite='None',
        )
        return response
    

class ChangeCoversation(GenericAPIView):
    """
    changes conversation cookie
    """
    serializer_class = ConversationSerializer

    def get(self, request: HttpRequest, id):
        conversation = Conversation.objects.filter(id=id).first()
        if not conversation:
            data = {
                "message": f"Conversation with id:{id} not found!"
            }
            return JsonResponse(
                data=data,
                status=status.HTTP_404_NOT_FOUND
            )

        data = self.serializer_class(conversation)

        response = JsonResponse(data=data)
        
        response.set_cookie(
            'conversation_id', 
            conversation.id,
            samesite='None',
        )
        return response


class GetConversationMessageList(ListAPIView):
    """
    gets conversation message
    """
    serializer_class = BaseMassegeSerializer

    def get_queryset(self):
        conversation_id = self.kwargs.get('conversation_id')

        if conversation_id is None:
            raise ValueError("conversation_id must be provided in the URL.")

        try:
            return Message.objects.filter(conversation_id=conversation_id)
        
        except ValueError:
            raise NotFound("Invalid conversation ID format.") 
        
        except Message.DoesNotExist:
            raise NotFound("Conversation not found.")
