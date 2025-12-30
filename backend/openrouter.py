"""OpenRouter API client for making LLM requests."""

import asyncio
import random
from typing import List, Dict, Any, Optional

import httpx

from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


def _sleep_backoff(attempt: int) -> float:
    return (0.6 * (2 ** attempt)) + (random.random() * 0.3)


def _print_resp_debug(model: str, resp: httpx.Response) -> None:
    # Helpful headers OpenRouter often returns
    rid = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
    rl_rem = resp.headers.get("x-ratelimit-remaining")
    rl_reset = resp.headers.get("x-ratelimit-reset")
    print(
        f"[OpenRouter] model={model} status={resp.status_code} "
        f"request_id={rid} ratelimit_remaining={rl_rem} ratelimit_reset={rl_reset}"
    )


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    max_retries: int = 2,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Returns:
        Dict {'content': str, 'reasoning_details': any} or None on failure.
    """
    # Debug: confirm key + model at runtime
    print(f"[OpenRouter] calling model={model} key_loaded={bool(OPENROUTER_API_KEY)}")

    if not OPENROUTER_API_KEY:
        print("[OpenRouter] OPENROUTER_API_KEY is missing. Check your .env")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Not strictly required, but recommended
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "llm-council",
    }

    payload: Dict[str, Any] = {
    "model": model,
    "messages": messages,
    "max_tokens": 256,  # ✅ prevents 402 with low credits
}


    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                )

                _print_resp_debug(model, resp)

                # Non-2xx: print body (this is the key info)
                if resp.status_code < 200 or resp.status_code >= 300:
                    print(f"[OpenRouter] body={resp.text}")

                    # retry on rate limit / transient server failures
                    if resp.status_code in (408, 409, 429) or 500 <= resp.status_code <= 599:
                        if attempt < max_retries:
                            await asyncio.sleep(_sleep_backoff(attempt))
                            continue
                    return None

                # Parse JSON
                try:
                    data = resp.json()
                except Exception as e:
                    print(f"[OpenRouter] model={model} JSON parse error: {repr(e)} body={resp.text}")
                    return None

                # Extract content safely
                try:
                    message = data["choices"][0]["message"]
                except Exception:
                    print(f"[OpenRouter] model={model} unexpected response shape: {data}")
                    return None

                content = (message.get("content") or "").strip()

                # 200 but empty content (happens sometimes)
                if not content:
                    print(f"[OpenRouter] model={model} EMPTY content (status=200). raw={data}")
                    if attempt < max_retries:
                        await asyncio.sleep(_sleep_backoff(attempt))
                        continue
                    return None

                return {
                    "content": content,
                    "reasoning_details": message.get("reasoning_details"),
                }

            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                print(f"[OpenRouter] model={model} timeout: {repr(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(_sleep_backoff(attempt))
                    continue
                return None

            except httpx.HTTPError as e:
                print(f"[OpenRouter] model={model} http error: {repr(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(_sleep_backoff(attempt))
                    continue
                return None

            except Exception as e:
                print(f"[OpenRouter] model={model} exception: {repr(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(_sleep_backoff(attempt))
                    continue
                return None

        return None

    finally:
        if owns_client and client is not None:
            await client.aclose()


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [query_model(model, messages, client=client) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for model, r in zip(models, results):
        if isinstance(r, Exception):
            print(f"[OpenRouter] model={model} task_exception={repr(r)}")
            out[model] = None
        else:
            out[model] = r
    return out
