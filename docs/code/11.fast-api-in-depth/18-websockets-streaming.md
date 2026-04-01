---
id: websockets-streaming
title: "18 · WebSockets & Streaming — Real-Time ML Output"
sidebar_label: "18 · WebSockets & Streaming"
sidebar_position: 18
tags: [websockets, sse, streaming, real-time, llm, intermediate]
---

# WebSockets & Streaming — Real-Time ML Output

> **Advanced Topic** — Streaming predictions, LLM token output, and real-time dashboards.

---

## Visual Reference

![WebSocket connection diagram](https://commons.wikimedia.org/wiki/Special:Redirect/file/Websocket_connection.png)

Source: [Wikimedia Commons - Websocket connection](https://commons.wikimedia.org/wiki/File:Websocket_connection.png)

## Why Streaming for ML?

Standard HTTP is **request-response**: client sends one request, server sends one response. This works for most APIs, but ML creates scenarios where you need data to flow continuously:

| Scenario | Why Request-Response Fails | Streaming Solution |
|----------|--------------------------|-------------------|
| LLM text generation | User waits 30s for full response | Stream tokens as they're generated |
| Model training progress | One response can't show 0%→100% | Push progress updates every second |
| Real-time fraud alerts | Server can't predict when fraud occurs | Server pushes alerts as they happen |
| Live model metrics dashboard | Metrics change every second | Server streams new values |

Two technologies solve this:
- **SSE (Server-Sent Events)**: One-way, server → client. Like a TV broadcast — you watch, server transmits.
- **WebSockets**: Two-way, either side can send at any time. Like a phone call — conversation flows both ways.

---

## SSE — Server-Sent Events (One-Way Streaming)

SSE is built on regular HTTP. The connection stays open and the server keeps sending data.

### LLM Token Streaming with SSE

```python title="app/routers/stream.py"
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/stream", tags=["streaming"])

class LLMRequest(BaseModel):
    prompt: str
    max_tokens: int = 200


async def simulate_llm_tokens(prompt: str):
    """
    Simulates an LLM generating text token by token.
    In a real system, this would call OpenAI/Anthropic/local model.
    """
    response_text = (
        f"Based on your question about '{prompt}', here is a detailed analysis. "
        f"Machine learning APIs are powerful tools that bridge the gap between "
        f"trained models and real-world applications. FastAPI makes this easier "
        f"by providing automatic validation, documentation, and high performance."
    )
    
    words = response_text.split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.05)  # 50ms between tokens — feels like typing


async def sse_generator(prompt: str, max_tokens: int):
    """
    Wraps token generator in SSE protocol format.
    
    SSE format (mandatory):
        data: <your JSON payload>\n\n
    
    The double newline (\n\n) is the event boundary.
    The browser's EventSource API parses this automatically.
    """
    token_count = 0
    
    async for token in simulate_llm_tokens(prompt):
        if token_count >= max_tokens:
            break
        
        # Each event: "data: {json}\n\n"
        event_data = json.dumps({"token": token, "done": False})
        yield f"data: {event_data}\n\n"
        token_count += 1
    
    # Final event tells the client generation is complete
    yield f"data: {json.dumps({'token': '', 'done': True, 'total_tokens': token_count})}\n\n"


@router.post("/llm")
async def stream_llm_response(request: LLMRequest):
    """
    Stream LLM output token by token.
    
    The client receives a continuous stream of SSE events.
    Each event contains one or a few tokens.
    
    Client-side (JavaScript):
    const response = await fetch('/stream/llm', {method: 'POST', body: ...});
    const reader = response.body.getReader();
    // loop: read chunks, parse "data: {...}\n\n", display token
    """
    return StreamingResponse(
        sse_generator(request.prompt, request.max_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # prevent caching of event stream
            "X-Accel-Buffering": "no",          # disable Nginx response buffering
            "Connection": "keep-alive",
        }
    )
```

### Consuming SSE in JavaScript

```javascript
// Frontend code (React/vanilla JS)
async function streamPrediction(prompt) {
    const response = await fetch('/stream/llm', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt, max_tokens: 200})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        // Chunk may contain multiple events
        const lines = chunk.split('\n\n').filter(Boolean);

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));

            if (data.done) {
                console.log('Stream complete. Total tokens:', data.total_tokens);
                break;
            }
            fullText += data.token;
            document.getElementById('output').textContent = fullText;
        }
    }
}
```

### SSE for Training Progress

```python
from celery.result import AsyncResult

@router.get("/training/{job_id}")
async def stream_training_progress(job_id: str):
    """
    Stream training job progress to the client.
    Client opens this SSE connection and gets updates every 2 seconds.
    """
    async def progress_generator():
        while True:
            result = AsyncResult(job_id)
            
            if result.state == "SUCCESS":
                data = json.dumps({
                    "status": "completed",
                    "result": result.result
                })
                yield f"data: {data}\n\n"
                break
            
            elif result.state == "FAILURE":
                data = json.dumps({"status": "failed", "error": str(result.info)})
                yield f"data: {data}\n\n"
                break
            
            else:
                meta = result.info if isinstance(result.info, dict) else {}
                data = json.dumps({
                    "status": "running",
                    "progress": meta.get("progress", 0),
                    "stage": meta.get("stage", "processing"),
                    "message": meta.get("message", "")
                })
                yield f"data: {data}\n\n"
            
            await asyncio.sleep(2)  # poll every 2 seconds
    
    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
```

---

## WebSockets — Bidirectional Real-Time Communication

WebSockets are different: both sides can send messages at any time. The connection stays open until explicitly closed.

### Basic WebSocket Endpoint

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    Real-time prediction endpoint via WebSocket.
    Client sends feature JSON, receives prediction JSON instantly.
    Stays connected for multiple predictions without reconnecting.
    """
    # Accept the connection (handshake)
    await websocket.accept()
    
    try:
        while True:
            # Wait for the client to send data
            # This blocks until data arrives (non-blocking in async context)
            raw_data = await websocket.receive_text()
            
            try:
                features = json.loads(raw_data)
                
                # Run your ML inference
                prediction = model.predict([list(features.values())])[0]
                confidence = float(max(model.predict_proba([list(features.values())])[0]))
                
                # Send the result back to the client
                await websocket.send_json({
                    "type": "prediction",
                    "prediction": str(prediction),
                    "confidence": confidence,
                })
            
            except (json.JSONDecodeError, KeyError) as e:
                await websocket.send_json({"type": "error", "message": str(e)})
    
    except WebSocketDisconnect:
        # Normal disconnection — client closed the browser tab, etc.
        print("Client disconnected")
    
    except Exception as e:
        # Unexpected error — send error message and close
        await websocket.send_json({"type": "error", "message": "Server error"})
        await websocket.close()
```

### WebSocket Connection Manager — Multi-Client Broadcasting

For scenarios where multiple clients need to receive the same real-time data (e.g., a fraud alert dashboard with multiple analysts):

```python title="app/core/ws_manager.py"
from fastapi import WebSocket
from typing import dict, list

class ConnectionManager:
    """
    Manages multiple WebSocket connections organized into "rooms".
    A room is just a named group of connections.
    
    Example rooms:
    - "fraud-alerts": all fraud analysts watching the dashboard
    - "training-room-123": data scientists monitoring experiment 123
    """
    
    def __init__(self):
        # room_name → list of active WebSocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, room: str):
        await websocket.accept()
        if room not in self.active_connections:
            self.active_connections[room] = []
        self.active_connections[room].append(websocket)
        print(f"Client connected to room '{room}'. Total: {len(self.active_connections[room])}")
    
    def disconnect(self, websocket: WebSocket, room: str):
        if room in self.active_connections:
            self.active_connections[room].discard(websocket)
    
    async def broadcast(self, room: str, data: dict):
        """Send data to ALL connected clients in a room."""
        if room not in self.active_connections:
            return
        
        # Copy the list — a client might disconnect during iteration
        connections = list(self.active_connections[room])
        dead_connections = []
        
        for websocket in connections:
            try:
                await websocket.send_json(data)
            except Exception:
                dead_connections.append(websocket)
        
        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(ws, room)

# Singleton — one manager for the entire application
manager = ConnectionManager()
```

```python title="app/routers/alerts.py"
from core.ws_manager import manager

@app.websocket("/ws/alerts/{room_name}")
async def websocket_alert_room(websocket: WebSocket, room_name: str):
    """
    WebSocket endpoint for real-time alert rooms.
    Multiple clients connect to the same room and all receive broadcasts.
    """
    await manager.connect(websocket, room=room_name)
    
    try:
        while True:
            # Keep connection alive
            # Any message from client is treated as a "heartbeat" acknowledgment
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, room=room_name)

# In your prediction endpoint, broadcast to all alert watchers:
@app.post("/predict")
async def predict(data: InsuranceInput):
    result = run_inference(data)
    
    if result.prediction == "high" and result.confidence > 0.9:
        # Broadcast to everyone in the "risk-alerts" room
        await manager.broadcast("risk-alerts", {
            "type": "high_risk_alert",
            "prediction": result.prediction,
            "confidence": result.confidence,
        })
    
    return result
```

---

## Q&A

**Q: When should I use SSE vs WebSockets?**

Use **SSE** when: the server pushes data to the client and the client never needs to send data back (LLM streaming, progress bars, live notifications). SSE is simpler, works over plain HTTP, and auto-reconnects on network drops.

Use **WebSockets** when: both sides need to send messages (real-time chat, collaborative editing, live gaming, interactive dashboards where clients send filter changes and receive filtered data).

**Q: How do I handle WebSocket connections across multiple Uvicorn workers?**

Each worker has its own `ConnectionManager` in memory. A broadcast from worker 1 only reaches clients connected to worker 1. For multi-worker setups, use Redis Pub/Sub as a shared channel: when one worker broadcasts, it publishes to Redis; all workers subscribe and forward to their local clients.

**Q: Can Nginx proxy WebSocket connections?**

Yes, but it requires specific configuration:
```nginx
location /ws/ {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;  # keep WebSocket connections alive for 1 hour
}
```

**Q: What's the maximum number of WebSocket connections?**

Each connection uses a file descriptor. Linux defaults to 1024 file descriptors per process. Increase with `ulimit -n 65535`. With asyncio, a single worker can practically handle 10,000+ concurrent idle WebSocket connections.
