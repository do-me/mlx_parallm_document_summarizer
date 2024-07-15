import asyncio
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
from starlette.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from mlx_parallm.utils import load, batch_generate, generate
from semantic_text_splitter import TextSplitter

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    model: str
    context: str
    range: Tuple[int, int]

generation_queues = {}

async def generate_stream(model, tokenizer, prompt, max_tokens, verbose, temp, request_id):
    queue = generation_queues[request_id]
    buffer = []
    for token in generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=verbose, temp=temp):
        buffer.append(token)
        if len(buffer) >= 10:
            await queue.put(''.join(buffer))
            buffer = []
        await asyncio.sleep(0.0000000000001)  # simulate a delay for streaming
    if buffer:
        await queue.put(''.join(buffer))
    await queue.put('[DONE]')
    del generation_queues[request_id]

@app.post("/process_data")
async def process_data(data: InputData, background_tasks: BackgroundTasks):
    try:
        print(data)
        splitter = TextSplitter(data.range)
        chunks = splitter.chunks(data.context)

        num_words = len(" ".join(chunks).split(" "))
        print(f"\n\n{len(chunks)} Chunks, {num_words} Words\n\n")
        prompts = [f"Summarize this document paragraph in one sentence focusing on its key messages:\n{chunk}" for chunk in chunks]
        model, tokenizer = load(data.model)
        responses = batch_generate(model, tokenizer, prompts=prompts, max_tokens=250, verbose=False, format_prompts=True, temp=0.0)
        master_prompt = "Prompt: Take a deep breath and write a concise document summary. Context:\n" + "\n".join([f"Partial Paragraph Summary: {i}" for i in responses])

        request_id = str(time.time())
        generation_queues[request_id] = asyncio.Queue()
        background_tasks.add_task(generate_stream, model, tokenizer, master_prompt, 500, True, 0.0, request_id)
        return {"request_id": request_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/stream/{request_id}")
async def stream(request: Request, request_id: str):
    if request_id not in generation_queues:
        raise HTTPException(status_code=404, detail="Request ID not found")
    queue = generation_queues[request_id]
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            token = await queue.get()
            if token == '[DONE]':
                break
            yield {"data": token}
    return EventSourceResponse(event_generator())

@app.get("/")
async def read_index():
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
