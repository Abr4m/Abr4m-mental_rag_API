"""
Services: QwenLLMPostProcessor
Concrete implementation of LLMPostProcessor using
Qwen/Qwen2.5-1.5B-Instruct via HuggingFace pipelines.
Falls back to a structured bullet list if the pipeline is unavailable.
Previously at infrastructure/ml/qwen_processor.py.
"""
import re
from typing import List

from app.models.interfaces import LLMPostProcessor

_PROMPT_TEMPLATE = """You are a helpful and compassionate mental health assistant.

Your task:
* Read the community suggestions below.
* Combine similar ideas.
* Produce a concise answer in 4-6 bullet points.
* Focus on practical, evidence-informed advice.
* Keep a supportive tone.
* Do not repeat the input text verbatim.
* Do not mention that you are an AI.
* Do not invent information not present in the suggestions.
* If appropriate, encourage seeking help from a qualified mental health professional.

User query:
{query}

Community suggestions:
{tips}

Final response:"""

_DISCLAIMER = (
    "\n\n---\n*These are community-sourced suggestions. "
    "Please consider speaking with a mental health professional "
    "if you need further support.*"
)


class QwenLLMPostProcessor(LLMPostProcessor):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 200,
    ) -> None:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        self._pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            device=device,
        )
        self._max_new_tokens = max_new_tokens

    def generate(self, query: str, suggestions: List[str]) -> str:
        tips_text = "\n".join(f"- {t[:300]}" for t in suggestions)
        prompt = _PROMPT_TEMPLATE.format(query=query, tips=tips_text)

        result = self._pipe(
            prompt,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"].strip()

        if not result.startswith("**"):
            result = "**Suggestions for you**\n\n" + result

        return result + _DISCLAIMER


class FallbackLLMPostProcessor(LLMPostProcessor):
    """
    Lightweight fallback used in tests or when the GPU/model is unavailable.
    Returns a plain bullet list without loading any model weights.
    """

    def generate(self, query: str, suggestions: List[str]) -> str:
        lines = [
            "Here are community-sourced suggestions that may help. "
            "These come from people with lived experience — "
            "please also consider speaking with a professional.\n"
        ]
        for text in suggestions:
            if len(text) > 350:
                sentences = re.split(r"(?<=[.!])\s+", text)
                text = " ".join(sentences[:3])
            lines.append(f"- {text}")
        return "\n".join(lines).strip()
