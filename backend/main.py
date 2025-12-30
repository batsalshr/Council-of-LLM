"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .council import (
    run_full_council_messages,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
)

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ----------------------------
# Models
# ----------------------------

class CreateConversationRequest(BaseModel):
    pass


class SendMessageRequest(BaseModel):
    content: str


class ConversationMetadata(BaseModel):
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(_: CreateConversationRequest):
    conversation_id = str(uuid.uuid4())
    return storage.create_conversation(conversation_id)


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


# ----------------------------
# NON-STREAMING MESSAGE
# ----------------------------

@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0

    # 1️⃣ Add user message
    storage.add_user_message(conversation_id, request.content)

    # Reload conversation with new message (now includes full history)
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2️⃣ Generate title if first message
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # 3️⃣ Run council using FULL conversation history ✅
    stage1, stage2, stage3, metadata = await run_full_council_messages(conversation["messages"])

    # 4️⃣ Store assistant response
    storage.add_assistant_message(conversation_id, stage1, stage2, stage3)

    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "metadata": metadata,
    }


# ----------------------------
# STREAMING MESSAGE (SSE)
# ----------------------------

@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Reload to get FULL message history
            conversation = storage.get_conversation(conversation_id)
            if conversation is None:
                raise Exception("Conversation not found after adding message")

            full_messages = conversation["messages"]
            latest_user_query = request.content

            # Build short conversation context (last 8 turns)
            context_lines = []
            for m in full_messages[-8:]:
                role = (m.get("role") or "").upper()
                content = (m.get("content") or "")
                context_lines.append(f"{role}: {content}")
            conversation_context = "\n".join(context_lines)

            # Title generation (async)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(latest_user_query))

            # ---- Stage 1 ----
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1 = await stage1_collect_responses(full_messages)  # ✅ full history
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1})}\n\n"

            # ---- Stage 2 ----
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2, label_to_model = await stage2_collect_rankings(
                latest_user_query,
                stage1,
                conversation_context=conversation_context,  # ✅ add context
            )
            aggregate = calculate_aggregate_rankings(stage2, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate}})}\n\n"

            # ---- Stage 3 ----
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3 = await stage3_synthesize_final(
                latest_user_query,
                stage1,
                stage2,
                conversation_context=conversation_context,  # ✅ add context
            )
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3})}\n\n"

            # Title update
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save assistant message
            storage.add_assistant_message(conversation_id, stage1, stage2, stage3)

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
