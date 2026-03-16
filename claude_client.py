"""
Claude AI client for DMC embroidery pattern assistant.
Provides pattern advice, color suggestions, and design help via Claude API.
"""

from __future__ import annotations

import os
from typing import Optional

import anthropic


def get_client() -> anthropic.Anthropic:
    """Create and return an Anthropic client using ANTHROPIC_API_KEY env var."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it to use Claude AI features."
        )
    return anthropic.Anthropic(api_key=api_key)


_SYSTEM_PROMPT = """Ты эксперт по вышиванию крестиком и DMC нитям.
Ты помогаешь пользователям создавать схемы для вышивания, подбирать цвета DMC,
и давать советы по технике вышивания. Отвечай на русском языке, если пользователь
пишет по-русски. Используй практические советы и конкретные номера DMC нитей.
Если пользователь спрашивает о схемах — рекомендуй подходящие размеры, количество цветов
и уровень сложности работы."""


def ask_claude(
    message: str,
    conversation_history: Optional[list[dict]] = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
) -> tuple[str, list[dict]]:
    """
    Send a message to Claude and get a response.

    Args:
        message: User message text
        conversation_history: Previous messages for multi-turn conversation.
                              Each dict has 'role' ('user'/'assistant') and 'content'.
        model: Claude model ID to use
        max_tokens: Maximum tokens in response

    Returns:
        Tuple of (response_text, updated_conversation_history)
    """
    client = get_client()

    if conversation_history is None:
        conversation_history = []

    # Add user message to history
    messages = conversation_history + [{"role": "user", "content": message}]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )

    response_text = response.content[0].text

    # Update history with both user and assistant turns
    updated_history = messages + [{"role": "assistant", "content": response_text}]

    return response_text, updated_history


def analyze_pattern_with_claude(
    pattern_json: str,
    user_question: str = "",
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    Ask Claude to analyze an embroidery pattern and provide advice.

    Args:
        pattern_json: JSON string of the pattern (from EmbroideryPattern.to_json())
        user_question: Optional specific question about the pattern
        model: Claude model ID

    Returns:
        Claude's analysis as a string
    """
    import json

    pattern_data = json.loads(pattern_json)
    num_colors = len(pattern_data.get("color_legend", {}))
    width = pattern_data.get("width", 0)
    height = pattern_data.get("height", 0)

    # Build a concise summary for Claude (not the full grid — too large)
    color_summary = []
    for dmc, info in list(pattern_data.get("color_legend", {}).items())[:20]:
        color_summary.append(
            f"DMC {dmc} ({info['name']}): {info['count']} крестиков"
        )

    prompt = f"""Я создал схему для вышивания крестиком со следующими параметрами:
- Размер: {width} × {height} крестиков
- Количество цветов DMC: {num_colors}
- Топ цвета:
{chr(10).join(color_summary[:10])}

{"Вопрос: " + user_question if user_question else "Дай общую оценку этой схемы и советы по работе с ней."}"""

    response, _ = ask_claude(prompt, model=model)
    return response


def suggest_pattern_improvements(
    description: str,
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    Ask Claude to suggest improvements or alternatives for a pattern description.

    Args:
        description: Text description of what user wants to embroider
        model: Claude model ID

    Returns:
        Claude's suggestions as a string
    """
    prompt = f"""Пользователь хочет создать схему для вышивания крестиком: {description}

Дай рекомендации:
1. Подходящий размер схемы (ширина × высота в крестиках)
2. Оптимальное количество цветов DMC
3. Уровень сложности
4. Советы по выбору ткани (Aida) и игл
5. Примерное время выполнения работы"""

    response, _ = ask_claude(prompt, model=model)
    return response
