"""Chat routes: grounded completion, streaming, and alias discovery.

``POST /chat`` and ``POST /chat/stream`` are separate routes rather than one route
with a ``stream: bool`` flag, because the desktop's ``request<T>()`` helper always
calls ``res.json()``. A flag would make one endpoint return two incompatible content
types and force a client-side special case; two routes keep the JSON path untouched.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from local_infra.api.app import get_chat
from local_infra.api.schemas import ChatRequest
from local_infra.llm.chat import ChatAnswer, ChatService
from local_infra.llm.types import ChatMessage, ChatUnavailable, StreamDelta

router = APIRouter(tags=["chat"])


def _history(body: ChatRequest) -> list[ChatMessage]:
    return [ChatMessage(t.role, t.content) for t in body.history]


@router.get("/chat/models")
async def chat_models(chat: ChatService = Depends(get_chat)) -> list[dict]:
    """Aliases the client may ask for, and whether each runs locally.

    Reports capability, never a secret: a configured cloud alias shows up as
    ``is_local: false``, but the key itself stays server-side.
    """
    return chat.aliases()


@router.post("/chat")
async def chat_completion(
    body: ChatRequest, chat: ChatService = Depends(get_chat)
) -> ChatAnswer:
    try:
        return await chat.ask(
            body.question,
            scope=body.scope,
            alias=body.alias,
            history=_history(body),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown alias {body.alias!r}") from None
    except ChatUnavailable as exc:
        # Every deployment failed. 503 with the reasons, not an opaque 500.
        raise HTTPException(status_code=503, detail=str(exc)) from None


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data))}\n\n"


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, chat: ChatService = Depends(get_chat)):
    """Server-sent events: ``token``* → ``usage`` → ``done``, or ``error``.

    Fallback is weaker here than on ``POST /chat``, deliberately: once a token has
    been sent the partial answer is already on the user's screen, so a mid-stream
    failure surfaces as an ``error`` event instead of silently switching models.
    """

    async def events() -> AsyncIterator[str]:
        try:
            async for item in chat.ask_stream(
                body.question,
                scope=body.scope,
                alias=body.alias,
                history=_history(body),
            ):
                if isinstance(item, StreamDelta):
                    yield _sse("token", {"delta": item.text})
                else:
                    yield _sse("usage", item)
            yield _sse("done", {})
        except KeyError:
            yield _sse("error", {"detail": f"unknown alias {body.alias!r}"})
        except ChatUnavailable as exc:
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            # Without this a reverse proxy will buffer the whole stream and the
            # client sees nothing until generation finishes — i.e. not streaming.
            "x-accel-buffering": "no",
        },
    )
