import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .llm import stream_answer
import asyncio


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        question = data.get("message", "")

        if question:
            try:
                async for token in stream_answer(
                    {"question": question}
                    ):
                    await self.send(text_data=json.dumps({"message": token}))
                    await asyncio.sleep(0.01) 

            except Exception as e:
                print(f"Error in consumer: {e}")
                await self.send(
                    text_data=json.dumps(
                        {"message": f"Error: {str(e)}"}
                    )
                )
