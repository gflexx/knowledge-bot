from django.db import models
from django.utils.timezone import now
import uuid


class ConversationManager(models.Manager):
    def check_conversation(request):
        if 'conversation_id' in request.COOKIES:
            conversation_id = Conversation.objects.filter(
                id=conversation_id
            ).first()
            if conversation_id:
                return conversation_id, False
            
            conversation_id = Conversation.objects.create()
            return conversation_id, True

        else:
            conversation_id = Conversation.objects.create()
            return conversation_id, True


class Conversation(models.Model):
    id = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False
    )
    title = models.TextField(
        blank=True,
        null=True
    )
    user = models.ForeignKey(
        "auth.User", 
        null=True, 
        blank=True, 
        on_delete=models.SET_NULL
    )
    creation_time = models.DateTimeField(
        auto_now_add=True,
        null=True, 
        blank=True, 
    )

    def __str__(self):
        return f"Conversation {self.id} ({'User: ' + self.user.username if self.user else 'Anonymous'})"


class Message(models.Model):
    conversation = models.ForeignKey(
        Conversation, 
        on_delete=models.CASCADE, 
        related_name="messages"
    )
    sender = models.CharField(
        max_length=10, 
        choices=[("user", "User"), ("bot", "Bot")]
    )
    content = models.TextField()
    timestamp = models.DateTimeField(default=now)

    def __str__(self):
        return f"{self.sender.capitalize()} ({self.timestamp})"