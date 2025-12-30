"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL


# ---------- helpers ----------

def _clip(text: str, limit: int = 6000) -> str:
    """Prevent giant prompts from breaking providers."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


# ---------- Stage 1 ----------

async def stage1_collect_responses(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.
    Uses full conversation messages so chat can continue.
    """
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    stage1_results: List[Dict[str, Any]] = []
    for model, response in responses.items():
        if response is not None:
            stage1_results.append(
                {
                    "model": model,
                    "response": (response.get("content") or "").strip(),
                }
            )

    return stage1_results


# ---------- Stage 2 ----------

async def stage2_collect_rankings(
    latest_user_query: str,
    stage1_results: List[Dict[str, Any]],
    conversation_context: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Only asks models that successfully responded in Stage 1.
    """
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, stage1_results)
    }

    responses_text = "\n\n".join(
        [f"Response {label}:\n{result['response']}" for label, result in zip(labels, stage1_results)]
    )

    context_block = ""
    if conversation_context.strip():
        context_block = f"\nConversation so far:\n{conversation_context}\n"

    ranking_prompt = f"""You are evaluating different responses to the user's latest question.

{context_block}
Latest Question: {latest_user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. Evaluate each response: what it does well and what it does poorly.
2. Then, at the very end, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
FINAL RANKING:
1. Response X
2. Response Y
...

- Start with "FINAL RANKING:" (all caps, with colon)
- Then list best to worst as a numbered list
- Each line must be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- No extra text in the ranking section.

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # ✅ Only ask models that answered in Stage 1
    active_models = [r["model"] for r in stage1_results]
    responses = await query_models_parallel(active_models, messages)

    stage2_results: List[Dict[str, Any]] = []
    for model, response in responses.items():
        if response is not None:
            full_text = (response.get("content") or "").strip()
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append(
                {
                    "model": model,
                    "ranking": full_text,
                    "parsed_ranking": parsed,
                }
            )

    return stage2_results, label_to_model


# ---------- Stage 3 ----------

async def stage3_synthesize_final(
    latest_user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    conversation_context: str = "",
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Uses short ranking summaries (parsed_ranking) to keep prompt size reasonable.
    """
    stage1_text = "\n\n".join(
        [f"Model: {r['model']}\nResponse:\n{_clip(r['response'], 4000)}" for r in stage1_results]
    )

    # ✅ Keep Stage 2 short (don’t send essays; send only parsed ranking)
    stage2_text = "\n\n".join(
        [
            f"Model: {r['model']}\nFINAL RANKING: {', '.join(r.get('parsed_ranking', []))}"
            for r in stage2_results
        ]
    )

    context_block = ""
    if conversation_context.strip():
        context_block = f"\nConversation so far:\n{conversation_context}\n"

    chairman_prompt = f"""You are the Chairman of an LLM Council.

{context_block}
Latest Question: {latest_user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings (summarized):
{stage2_text}

Task:
Synthesize a single, accurate, helpful final answer to the user's latest question.
Use the best parts of the responses, resolve disagreements, and avoid errors.
Write clearly and directly:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    response = await query_model(CHAIRMAN_MODEL, messages)

    content = (response or {}).get("content", "")
    content = (content or "").strip()

    if not content:
        # ✅ Fallback: return first available Stage 1 response
        fallback = stage1_results[0]["response"] if stage1_results else ""
        return {
            "model": CHAIRMAN_MODEL,
            "response": fallback if fallback else "Error: Unable to generate final synthesis.",
        }

    return {"model": CHAIRMAN_MODEL, "response": content}


# ---------- ranking parsing + aggregates ----------

def parse_ranking_from_text(ranking_text: str) -> List[str]:
    import re

    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            numbered_matches = re.findall(r"\d+\.\s*Response [A-Z]", ranking_section)
            if numbered_matches:
                out = []
                for m in numbered_matches:
                    mm = re.search(r"Response [A-Z]", m)
                    if mm:
                        out.append(mm.group())
                return out

            return re.findall(r"Response [A-Z]", ranking_section)

    return re.findall(r"Response [A-Z]", ranking_text)


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    from collections import defaultdict

    model_positions = defaultdict(list)

    for ranking in stage2_results:
        parsed_ranking = ranking.get("parsed_ranking") or parse_ranking_from_text(ranking.get("ranking", ""))

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    aggregate: List[Dict[str, Any]] = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append(
                {"model": model, "average_rank": round(avg_rank, 2), "rankings_count": len(positions)}
            )

    aggregate.sort(key=lambda x: x["average_rank"])
    return aggregate


# ---------- titles ----------

async def generate_conversation_title(user_query: str) -> str:
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        return "New Conversation"

    title = (response.get("content") or "New Conversation").strip()
    title = title.strip("\"'")
    if len(title) > 50:
        title = title[:47] + "..."
    return title


# ---------- public API ----------

async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Backwards compatible: single-turn input.
    """
    messages = [{"role": "user", "content": user_query}]
    return await run_full_council_messages(messages)


async def run_full_council_messages(messages: List[Dict[str, str]]) -> Tuple[List, List, Dict, Dict]:
    """
    Multi-turn: pass full conversation history to continue chat.
    """
    latest_user_query = messages[-1]["content"] if messages else ""

    # Add a small context summary (last 8 turns)
    context_lines = []
    for m in messages[-8:]:
        role = (m.get("role") or "").upper()
        content = (m.get("content") or "")
        context_lines.append(f"{role}: {content}")
    conversation_context = _clip("\n".join(context_lines), 3000)

    stage1_results = await stage1_collect_responses(messages)

    if not stage1_results:
        return [], [], {"model": "error", "response": "All models failed to respond. Please try again."}, {}

    stage2_results, label_to_model = await stage2_collect_rankings(
        latest_user_query,
        stage1_results,
        conversation_context=conversation_context,
    )

    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    stage3_result = await stage3_synthesize_final(
        latest_user_query,
        stage1_results,
        stage2_results,
        conversation_context=conversation_context,
    )

    metadata = {"label_to_model": label_to_model, "aggregate_rankings": aggregate_rankings}
    return stage1_results, stage2_results, stage3_result, metadata
