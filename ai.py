import os
import json
import re
import requests

MODEL = "gemini-2.5-flash"
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def _call(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_KEY")

    if not api_key:
        raise Exception("GEMINI_KEY not found in environment variables")

    url = f"{API_ENDPOINT}?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    # # Print response for debugging
    # print("Gemini response:", response.text)

    response.raise_for_status()

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise Exception("Unexpected Gemini response format: " + json.dumps(data))


def _clean_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_questions(topic: str, difficulty: str, count: int = 5) -> list:
    prompt = f"""
Generate {count} multiple-choice quiz questions about "{topic}" at {difficulty} difficulty.

Return ONLY a JSON array — no markdown, no extra text.

Each element must contain:
"id": integer (1-based)
"topic": "{topic}"
"question": string
"options": array of exactly 4 strings
"answerIndex": integer 0-3 (index of the correct option)
"explanation": one-sentence explanation of the correct answer
"""

    result = _call(prompt)
    cleaned = _clean_json(result)
    return json.loads(cleaned)


def explain_answer(question: str, options: list, answer_index: int, selected_index: int) -> str:
    prompt = f"""
A student answered a quiz question incorrectly.

Question: {question}

Options: {json.dumps(options)}

Correct answer:
Option {answer_index} — "{options[answer_index]}"

Student chose:
Option {selected_index} — "{options[selected_index]}"

Explain in 2-3 sentences why the correct answer is right and why the student's choice is wrong.
"""

    return _call(prompt).strip()


def recommend_study(topic_scores: list) -> str:
    prompt = f"""
A student just completed a quiz. Results by topic:

{json.dumps(topic_scores, indent=2)}

Each entry has:
topic
correct
total

Write a short personalised study recommendation (3–4 sentences) focusing on their weakest areas.
"""

    return _call(prompt).strip()