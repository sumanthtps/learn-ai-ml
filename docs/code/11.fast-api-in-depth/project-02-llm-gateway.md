---
id: project-llm-gateway
title: "Project 2 · Multi-Provider LLM API Gateway"
sidebar_label: "🟡 Project 2 · LLM Gateway"
sidebar_position: 22
tags: [project, llm, openai, streaming, gateway, rate-limiting, cost-tracking, advanced]
---

# Project 2 · Multi-Provider LLM API Gateway

> **Industry-Grade Project** — Build a unified LLM API that routes to multiple providers, streams tokens, tracks costs, and rate-limits by user.

---

## What Problem Does This Solve?

Imagine your company uses LLMs across multiple products:
- A customer support chatbot (needs cheap, fast responses)
- A code review tool (needs GPT-4o quality)
- A document summarizer (needs large context window)

Without a gateway, every team hardcodes OpenAI credentials, manages their own rate limits, and has no visibility into costs. When OpenAI has an outage, everything breaks.

**The gateway solves all of this:**
- One unified endpoint for all LLM calls
- Automatic routing to the right provider
- Fallback to backup when primary is down
- Token tracking and cost reporting per team
- Rate limiting prevents runaway bills
- Streaming for responsive UIs

---

## Architecture Understanding

```
Your Internal Services          LLM Gateway              External Providers
──────────────────────          ───────────              ──────────────────
                                                          
Customer Support Bot ──POST──►  ┌───────────┐  ──────►  OpenAI GPT-4o
                                │  Gateway  │
Code Review Tool ─────POST──►   │           │  ──────►  Anthropic Claude
                                │  • Auth   │
Document Summarizer ──POST──►   │  • Route  │  ──────►  Ollama (local)
                                │  • Limit  │
                                │  • Cache  │
                                │  • Track  │
                                │  • Log    │
                                └───────────┘
                                     │
                                     ▼
                               PostgreSQL + Redis
                               (costs, usage, cache)
```

The gateway is **OpenAI API-compatible** — clients use the exact same OpenAI SDK without any changes. The gateway handles routing transparently.

---

## Step 1: The Provider Abstraction — Strategy Pattern

The **Strategy Pattern** is a design pattern where you define an interface (abstract class), then create interchangeable implementations. This lets you swap providers without changing the rest of your code.

```python title="app/providers/base.py"
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from pydantic import BaseModel

class Message(BaseModel):
    role: str   # "system", "user", or "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False
    system: str | None = None   # Anthropic-style system prompt

class ChatChunk(BaseModel):
    """One piece of the streaming output."""
    content: str
    finish_reason: str | None = None
    input_tokens: int = 0   # only in final chunk
    output_tokens: int = 0  # only in final chunk

class LLMProvider(ABC):
    """
    Abstract base class that all LLM providers must implement.
    
    Why abstract? It forces every provider (OpenAI, Anthropic, Ollama)
    to implement the same interface. The gateway code can call
    provider.complete(request) without knowing which provider it is.
    
    This is called the Liskov Substitution Principle:
    "Objects of a subclass should be substitutable for objects of the superclass."
    """
    name: str                    # "openai", "anthropic", "ollama"
    supported_models: list[str]  # models this provider handles
    
    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatChunk:
        """Make a non-streaming completion. Returns the full response."""
        ...
    
    @abstractmethod
    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        """Stream the completion token by token."""
        ...
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is reachable (health check)."""
        ...
```

```python title="app/providers/openai_provider.py"
import httpx
import json
from typing import AsyncGenerator
from .base import LLMProvider, ChatRequest, ChatChunk

class OpenAIProvider(LLMProvider):
    """
    Implements LLMProvider for OpenAI's API.
    
    Uses httpx (async HTTP client) to call OpenAI's REST API.
    We don't use the openai library to maintain control over
    streaming behavior and error handling.
    """
    name = "openai"
    supported_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    BASE_URL = "https://api.openai.com/v1"
    
    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    def _build_payload(self, request: ChatRequest) -> dict:
        """Convert our unified ChatRequest to OpenAI's format."""
        messages = [m.model_dump() for m in request.messages]
        if request.system:
            # Prepend system message (OpenAI uses "system" role)
            messages.insert(0, {"role": "system", "content": request.system})
        
        return {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
    
    async def complete(self, request: ChatRequest) -> ChatChunk:
        """Call OpenAI synchronously (get full response at once)."""
        payload = self._build_payload(request)
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        
        return ChatChunk(
            content=data["choices"][0]["message"]["content"],
            finish_reason=data["choices"][0]["finish_reason"],
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
        )
    
    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        """
        Call OpenAI with streaming enabled.
        
        OpenAI returns Server-Sent Events. Each event looks like:
        data: {"choices": [{"delta": {"content": "Hello"}}]}
        
        We yield one ChatChunk per event.
        """
        payload = {**self._build_payload(request), "stream": True}
        
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # remove "data: " prefix
                    if data_str == "[DONE]":
                        break  # OpenAI signals end of stream with [DONE]
                    
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"]
                    
                    if "content" in delta and delta["content"]:
                        yield ChatChunk(
                            content=delta["content"],
                            finish_reason=data["choices"][0].get("finish_reason"),
                        )
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{self.BASE_URL}/models",
                    headers=self.headers
                )
                return r.status_code == 200
        except Exception:
            return False
```

---

## Step 2: Provider Router with Automatic Fallback

```python title="app/services/router_service.py"
import logging
from providers.base import LLMProvider, ChatRequest, ChatChunk

logger = logging.getLogger(__name__)

# Which provider handles which models
MODEL_ROUTING = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-3.5-turbo": "openai",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-haiku-20240307": "anthropic",
}

# If primary fails, try this provider instead
FALLBACK_CHAIN = {
    "openai": "anthropic",
    "anthropic": "openai",
}

# Equivalent models on fallback provider (same capability tier)
FALLBACK_MODELS = {
    "anthropic": "claude-3-haiku-20240307",    # fast Anthropic model
    "openai": "gpt-4o-mini",                   # fast OpenAI model
}

class ProviderRouter:
    """
    Routes requests to the right LLM provider and handles failover.
    
    Failover means: if the primary provider is down or returns an error,
    automatically retry with the backup provider. The client never knows
    which provider actually served their request.
    """
    
    def __init__(self, providers: dict[str, LLMProvider]):
        self.providers = providers
    
    def get_provider(self, model: str) -> LLMProvider:
        provider_name = MODEL_ROUTING.get(model)
        if not provider_name:
            raise ValueError(f"Unsupported model: '{model}'")
        return self.providers[provider_name]
    
    async def complete_with_fallback(
        self,
        request: ChatRequest
    ) -> tuple[ChatChunk, str]:
        """
        Try the primary provider. If it fails, try the fallback.
        Returns (result, provider_name_that_succeeded).
        """
        primary = self.get_provider(request.model)
        
        try:
            result = await primary.complete(request)
            logger.info(f"Completed via {primary.name}, model={request.model}")
            return result, primary.name
        
        except Exception as primary_error:
            logger.warning(
                f"Primary provider {primary.name} failed: {primary_error}. "
                f"Trying fallback..."
            )
            
            fallback_name = FALLBACK_CHAIN.get(primary.name)
            if not fallback_name or fallback_name not in self.providers:
                raise primary_error  # no fallback available
            
            fallback = self.providers[fallback_name]
            fallback_model = FALLBACK_MODELS[fallback_name]
            fallback_request = request.model_copy(update={"model": fallback_model})
            
            try:
                result = await fallback.complete(fallback_request)
                logger.info(f"Fallback succeeded via {fallback.name}")
                return result, fallback.name
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise fallback_error  # both failed

router_service: ProviderRouter = None  # initialized at startup
```

---

## Step 3: Token Cost Tracking

```python title="app/services/cost_service.py"
from decimal import Decimal

# Pricing per 1M tokens (USD) — update when providers change pricing
PRICING = {
    "gpt-4o":                    {"input": Decimal("2.50"),  "output": Decimal("10.00")},
    "gpt-4o-mini":               {"input": Decimal("0.15"),  "output": Decimal("0.60")},
    "gpt-3.5-turbo":             {"input": Decimal("0.50"),  "output": Decimal("1.50")},
    "claude-3-5-sonnet-20241022":{"input": Decimal("3.00"),  "output": Decimal("15.00")},
    "claude-3-haiku-20240307":   {"input": Decimal("0.25"),  "output": Decimal("1.25")},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """
    Calculate the cost of an API call in USD.
    
    Example:
    GPT-4o with 500 input tokens + 200 output tokens:
    = (500/1_000_000 × $2.50) + (200/1_000_000 × $10.00)
    = $0.00125 + $0.002
    = $0.00325
    
    We use Decimal instead of float to avoid floating-point rounding errors
    when summing many small costs over time.
    """
    if model not in PRICING:
        return Decimal("0")
    
    prices = PRICING[model]
    cost = (
        Decimal(input_tokens) / Decimal("1_000_000") * prices["input"]
        + Decimal(output_tokens) / Decimal("1_000_000") * prices["output"]
    )
    return cost.quantize(Decimal("0.000001"))  # round to 6 decimal places
```

---

## Step 4: Rate Limiting by User Tier

```python title="app/core/rate_limiter.py"
import redis.asyncio as aioredis
from fastapi import HTTPException
from datetime import datetime, timezone

# Different tiers get different limits
TIER_LIMITS = {
    "free":       {"rpm": 10,   "tokens_per_day": 100_000},
    "starter":    {"rpm": 60,   "tokens_per_day": 1_000_000},
    "pro":        {"rpm": 300,  "tokens_per_day": 10_000_000},
    "enterprise": {"rpm": 3000, "tokens_per_day": None},  # unlimited
}

class RateLimiter:
    """
    Per-user rate limiting using Redis.
    
    We use two separate limits:
    1. RPM (requests per minute): prevents request flooding
    2. Tokens per day: prevents cost overruns
    
    Both use Redis atomic operations (INCR + EXPIRE) which are thread-safe
    and work correctly even with multiple API instances.
    """
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def check_rpm(self, user_id: str, tier: str):
        """Check and increment requests-per-minute counter."""
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        now = datetime.now(timezone.utc)
        
        # Key includes the current minute — automatically resets every minute
        key = f"rpm:{user_id}:{now.strftime('%Y%m%d%H%M')}"
        
        # INCR is atomic: increments and returns new value in one operation
        # This prevents race conditions where two requests both read "0"
        # and both think they're the first request
        current = await self.redis.incr(key)
        await self.redis.expire(key, 70)  # expire after 70 seconds (bit of slack)
        
        if current > limits["rpm"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "limit": limits["rpm"],
                    "window": "per_minute",
                    "retry_after_seconds": 60
                },
                headers={"Retry-After": "60"}
            )
    
    async def check_and_record_tokens(self, user_id: str, tier: str, tokens: int):
        """Check daily token limit and record usage."""
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        if limits["tokens_per_day"] is None:
            return  # unlimited tier
        
        now = datetime.now(timezone.utc)
        key = f"tokens:{user_id}:{now.strftime('%Y%m%d')}"  # daily key
        
        new_total = await self.redis.incrby(key, tokens)
        await self.redis.expire(key, 86400)  # expire after 24 hours
        
        if new_total > limits["tokens_per_day"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "daily_token_limit_exceeded",
                    "limit": limits["tokens_per_day"],
                    "used": new_total,
                }
            )
```

---

## Step 5: The Streaming Completions Endpoint

```python title="app/routers/completions.py"
import json
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from providers.base import ChatRequest
from services.router_service import router_service
from services.cost_service import calculate_cost

router = APIRouter(prefix="/v1", tags=["completions"])

@router.post("/chat/completions")
async def chat_completions(
    request: ChatRequest,
    http_request: Request,
    current_user=Depends(get_current_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """
    OpenAI-compatible chat completions endpoint.
    
    Clients can use the official OpenAI Python SDK with just a changed base_url:
    
        from openai import OpenAI
        client = OpenAI(
            api_key="your-gateway-key",
            base_url="https://your-gateway.com/v1"
        )
        # Now works transparently with any model your gateway supports!
    """
    request_id = str(uuid.uuid4())
    
    # Check rate limit before doing any work
    await rate_limiter.check_rpm(str(current_user.id), current_user.tier)
    
    if not request.stream:
        # ─── Non-streaming path ───────────────────────────────────
        chunk, provider_used = await router_service.complete_with_fallback(request)
        cost = calculate_cost(request.model, chunk.input_tokens, chunk.output_tokens)
        
        # Record token usage (for daily limit + billing)
        await rate_limiter.check_and_record_tokens(
            str(current_user.id),
            current_user.tier,
            chunk.input_tokens + chunk.output_tokens
        )
        
        # Log to database for billing report
        await log_usage(
            user_id=current_user.id,
            model=request.model,
            provider=provider_used,
            input_tokens=chunk.input_tokens,
            output_tokens=chunk.output_tokens,
            cost_usd=float(cost),
        )
        
        # Return in OpenAI's response format (for SDK compatibility)
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": chunk.content},
                "finish_reason": chunk.finish_reason,
            }],
            "usage": {
                "prompt_tokens": chunk.input_tokens,
                "completion_tokens": chunk.output_tokens,
                "total_tokens": chunk.input_tokens + chunk.output_tokens,
            },
            "provider": provider_used,   # extra field showing which provider was used
        }
    
    # ─── Streaming path ───────────────────────────────────────────
    provider = router_service.get_provider(request.model)
    
    async def streaming_generator():
        """
        Yields Server-Sent Events in OpenAI's streaming format.
        The OpenAI SDK handles this format automatically.
        """
        total_tokens = 0
        
        async for chunk in provider.stream(request):
            total_tokens += len(chunk.content.split())  # rough estimate
            
            event_data = json.dumps({
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk.content},
                    "finish_reason": chunk.finish_reason,
                }],
            })
            yield f"data: {event_data}\n\n"
        
        # Final DONE event
        yield "data: [DONE]\n\n"
        
        # Log approximate usage (we don't have exact token counts in streaming)
        await log_usage(
            user_id=current_user.id,
            model=request.model,
            provider=provider.name,
            output_tokens=total_tokens,
        )
    
    return StreamingResponse(
        streaming_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id}
    )
```

---

## Using the Gateway

```python
# Existing code using OpenAI directly:
from openai import OpenAI
client = OpenAI(api_key="sk-openai-key")

# Switching to your gateway — only two lines change:
from openai import OpenAI
client = OpenAI(
    api_key="your-gateway-api-key",  # ← your gateway key
    base_url="https://your-gateway.com/v1"  # ← your gateway URL
)

# All existing code works unchanged!
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",  # ← can now use Anthropic models too!
    messages=[{"role": "user", "content": "Explain transformers"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Key Learnings From This Project

| Concept | Where Applied |
|---------|--------------|
| Strategy Pattern (polymorphism) | `LLMProvider` ABC, `OpenAIProvider`, `AnthropicProvider` |
| Async HTTP with httpx | `async with httpx.AsyncClient().stream(...)` |
| SSE streaming responses | `StreamingResponse` + async generator |
| Automatic provider fallback | `complete_with_fallback()` with try/except |
| Redis rate limiting | Atomic `INCR` with minute-based keys |
| Cost tracking with Decimal | Avoid float precision errors on tiny amounts |
| OpenAI-compatible API | Clients use OpenAI SDK unchanged |
| JWT authentication | Bearer tokens checked on every request |
