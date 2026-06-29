from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import AzureOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic_search import semantic_search  # noqa: E402


OUTPUT_PATH = ROOT_DIR / "evaluation" / "rag_evaluation.json"
TRANSCRIPT_PATH = ROOT_DIR / "evaluation" / "rag_chat_transcript.txt"


SYSTEM_MESSAGE = """
Du bist ein hilfreicher Produkt-Chatbot für einen Retail-Prototyp.
Nutze ausschließlich die bereitgestellten Produktinformationen aus dem Retrieval-Kontext.

Regeln:
- Antworte immer auf Deutsch.
- Erfinde keine Produkte, Zutaten, Kategorien, Allergene, Preise oder Verfügbarkeiten.
- Wenn eine Information nicht im Kontext enthalten ist, sage das klar.
- Nenne nur Produkteigenschaften, die im Kontext stehen.
- Antworte knapp, aber vollständig.
""".strip()


TEST_DIALOGS: List[Dict[str, Any]] = [
    {
        "id": "rag_01_apfelsaft_exotisch",
        "description": "Mehrstufiger Dialog zu Apfelsaft und exotischer Frucht",
        "turns": [
            "Ich möchte einen Apfelsaft trinken. Welcher steht zur Verfügung?",
            "Ich hätte gerne einen Apfelsaft mit einer exotischen Frucht. Welchen würden Sie mir empfehlen?",
        ],
    },
    {
        "id": "rag_02_veganes_fruehstueck",
        "description": "Produktempfehlung für veganes Frühstück",
        "turns": [
            "Welche veganen Frühstücksprodukte gibt es?"
        ],
    },
    {
        "id": "rag_03_proteinreiche_produkte",
        "description": "Produktempfehlung für proteinreiche Produkte",
        "turns": [
            "Ich suche proteinreiche Produkte. Was kannst du empfehlen?"
        ],
    },
    {
        "id": "rag_04_bio_produkte",
        "description": "Suche nach Bio-Produkten",
        "turns": [
            "Gibt es Bio-Produkte?"
        ],
    },
    {
        "id": "rag_05_schokolade",
        "description": "Suche nach Schokoladenprodukten",
        "turns": [
            "Welche Schokolade gibt es?"
        ],
    },
]


def get_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    missing = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")

    if missing:
        raise RuntimeError(
            "Fehlende Azure/OpenAI Environment Variables: "
            + ", ".join(missing)
            + "\nSetze sie vorher mit export ..."
        )

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def ask_rag(
    client: AzureOpenAI,
    deployment: str,
    user_text: str,
    history: List[Dict[str, str]],
    k: int = 5,
) -> Dict[str, Any]:
    retrieval_raw = semantic_search(user_text, k=k)
    retrieval = json.loads(retrieval_raw)

    context = json.dumps(retrieval, ensure_ascii=False, indent=2)

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                "Nutzerfrage:\n"
                f"{user_text}\n\n"
                "Retrieval-Kontext aus der Produktsuche:\n"
                f"{context}\n\n"
                "Beantworte die Nutzerfrage ausschließlich auf Basis dieses Kontexts."
            ),
        }
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=700,
    )

    assistant_text = response.choices[0].message.content or ""

    return {
        "user": user_text,
        "retrieval": retrieval,
        "assistant": assistant_text,
    }


def build_transcript(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("RAG Evaluation Transcript")
    lines.append(f"Generated at: { datetime.now().isoformat(timespec='seconds') }")
    lines.append("")

    for dialog in results:
        lines.append("=" * 80)
        lines.append(f"Dialog: {dialog['id']}")
        lines.append(f"Beschreibung: {dialog.get('description', '')}")
        lines.append("")

        for i, turn in enumerate(dialog["turn_results"], start=1):
            lines.append(f"Turn {i}")
            lines.append(f"User: {turn['user']}")
            lines.append("")
            lines.append("Retrieval Top-5:")
            for rank, item in enumerate(turn["retrieval"].get("results", []), start=1):
                title = item.get("title")
                category = item.get("category")
                score = item.get("score")
                ingredients = item.get("ingredients")
                lines.append(
                    f"{rank}. {title} | Kategorie: {category} | Score: {score} | Zutaten: {ingredients}"
                )
            lines.append("")
            lines.append(f"Assistant: {turn['assistant']}")
            lines.append("")

        lines.append("Bewertung:")
        evaluation = dialog.get("manual_evaluation", {})
        lines.append(f"- Retrieval-Korrektheit: {evaluation.get('retrieval_correctness', 'TODO')}")
        lines.append(f"- Faktentreue: {evaluation.get('faithfulness', 'TODO')}")
        lines.append(f"- Halluzination: {evaluation.get('hallucination', 'TODO')}")
        lines.append(f"- Kommentar: {evaluation.get('comment', 'TODO')}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    client = get_client()

    results: List[Dict[str, Any]] = []

    for dialog in TEST_DIALOGS:
        print(f"\n=== {dialog['id']} ===")

        history: List[Dict[str, str]] = []
        turn_results: List[Dict[str, Any]] = []

        for user_text in dialog["turns"]:
            print(f"User: {user_text}")
            turn_result = ask_rag(
                client=client,
                deployment=deployment,
                user_text=user_text,
                history=history,
                k=5,
            )

            print(f"Bot: {turn_result['assistant']}\n")

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": turn_result["assistant"]})
            turn_results.append(turn_result)

        results.append(
            {
                "id": dialog["id"],
                "description": dialog["description"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "turn_results": turn_results,
                "manual_evaluation": {
                    "retrieval_correctness": "TODO: korrekt / teilweise korrekt / nicht korrekt",
                    "faithfulness": "TODO: korrekt / teilweise korrekt / nicht korrekt",
                    "hallucination": "TODO: nein / gering / ja",
                    "comment": "TODO",
                },
            }
        )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        TRANSCRIPT_PATH.write_text(
            build_transcript(results),
            encoding="utf-8",
        )

    print("\nFertig.")
    print(f"JSON gespeichert unter: {OUTPUT_PATH}")
    print(f"Transcript gespeichert unter: {TRANSCRIPT_PATH}")


if __name__ == "__main__":
    main()