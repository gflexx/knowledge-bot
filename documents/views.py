from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView
import json
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import sync_and_async_middleware
import asyncio

from .llm import stream_answer
from .serializers import *


@csrf_exempt
@sync_and_async_middleware
async def answer_question(request):
    """
    streams answer from Google Gemini API.
    """
    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST requests are allowed"}, 
            status=405
        )

    try:
        data = json.loads(request.body)
        question = data.get("question")
        if not question:
            return JsonResponse(
                {"error": "Question is required"}, 
                status=400
            )
        
    except json.JSONDecodeError:
        return JsonResponse(
            {"error": "Invalid JSON format"}, 
            status=400
        )
    
    async def token_generator():
        try:
            async for token in stream_answer({"question": question}):
                yield token
                await asyncio.sleep(0.01)
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingHttpResponse(token_generator(), content_type='text/plain')
    

class DocumentListCreateAPiView(ListCreateAPIView):
    """
    lists and creates documents
    """
    serializer_class = DocumentSerializer
    queryset = Document.objects.all()


class DocumentUpdateDelete(RetrieveUpdateDestroyAPIView):
    """
    updates deletes documents
    """
    serializer_class = DocumentSerializer
    queryset = Document.objects.all()
    lookup_field = 'id'