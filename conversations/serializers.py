from rest_framework import serializers

from .models import *


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = "__all__"


class BaseMassegeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = "__all__"


class MessageSerializer(BaseMassegeSerializer):
    class Meta:
        model = Message
        fields = "__all__"

    def to_representation(self, instance):
        self.fields["conversation"] = ConversationSerializer(read_only=True)
        return super(MessageSerializer, self).to_representation(instance)