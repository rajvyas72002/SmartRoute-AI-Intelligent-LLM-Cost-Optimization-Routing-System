"""Additive-score complexity estimator. Each rule that fires adds points AND a human-readable reason."""
import re
from core.schemas import Complexity, TaskCategory as T, TaskFeatures
from core.task_classifier import classify_task, count_subtasks

_REASONING = re.compile(r"\b(step[- ]by[- ]step|prove|why|trade-?offs?|compare|design|optimi[sz]e|derive|edge cases?|time complexity|justify|analy[sz]e)\b", re.I)
_CATEGORY_POINTS = {T.MATH: 2, T.PLANNING: 2, T.MULTI_PART: 1, T.CODING: 1, T.DATA_ANALYSIS: 1}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)      # documented approximation: ~4 chars/token


def analyze(prompt: str, simple_max: int = 1, moderate_max: int = 3) -> TaskFeatures:
    category = classify_task(prompt)
    words = len(prompt.split())
    subtasks = count_subtasks(prompt)
    score, reasons = 0, []

    if words > 300:
        score += 2; reasons.append(f"long prompt ({words} words) +2")
    elif words > 120:
        score += 1; reasons.append(f"medium-long prompt ({words} words) +1")

    pts = _CATEGORY_POINTS.get(category, 0)
    if pts:
        score += pts; reasons.append(f"category '{category.value}' +{pts}")

    if subtasks >= 3:
        score += 2; reasons.append(f"{subtasks} subtasks +2")
    elif subtasks == 2:
        score += 1; reasons.append("2 subtasks +1")

    hits = len(_REASONING.findall(prompt))
    if hits:
        p = min(hits, 2)
        score += p; reasons.append(f"{hits} reasoning indicator(s) +{p}")

    if not reasons:
        reasons.append("no complexity signals fired")

    level = Complexity.SIMPLE if score <= simple_max else Complexity.MODERATE if score <= moderate_max else Complexity.COMPLEX
    return TaskFeatures(category=category, complexity=level, score=score, subtask_count=subtasks,
                        word_count=words, est_input_tokens=estimate_tokens(prompt), reasons=reasons)
