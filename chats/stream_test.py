import asyncio
from chats.llm import stream_answer

async def test_stream():
    async for token in stream_answer({"question": "What services do you handle?"}):
        print("Response:", token["response"])
        print("Sources:", token["sources"])

asyncio.run(test_stream())
