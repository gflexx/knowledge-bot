from django.urls import path
from .views import *

urlpatterns = [
    path('check/', CheckConversation.as_view()),
    path('start/', StartConversation.as_view()),
    path('change/<uuid:conversation_id>/', ChangeCoversation.as_view()),
    path('masseges/get/<uuid:conversation_id>/', GetConversationMessageList.as_view())
]