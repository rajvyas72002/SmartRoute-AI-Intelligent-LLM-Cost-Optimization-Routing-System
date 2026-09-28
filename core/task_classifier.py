"""Rule-based task classifier. Interpretable on purpose: every decision is a regex you can read."""
import re
from core.schemas import TaskCategory as T

_RULES: list[tuple[T, re.Pattern]] = [
    (T.SUMMARIZATION, re.compile(r"\b(summari[sz]e|tl;?dr|summary|condense)\b", re.I)),
    (T.REWRITING, re.compile(r"\b(rewrite|rephrase|paraphrase|proofread|fix (the )?grammar|make .* (formal|shorter))\b", re.I)),
    (T.CODING, re.compile(r"\b(python|code|function|class|bug|debug|script|sql|javascript|leetcode|algorithm|implement)\b|```", re.I)),
    (T.MATH, re.compile(r"\b(solve|calculate|equation|integral|derivative|probability|prove|theorem)\b|\d\s*[\+\-\*/\^]\s*\d", re.I)),
    (T.DATA_ANALYSIS, re.compile(r"\b(dataset|dataframe|pandas|csv|correlation|regression|analy[sz]e (the )?data|statistics)\b", re.I)),
    (T.PLANNING, re.compile(r"\b(plan|roadmap|strategy|schedule|step-by-step plan|design a)\b", re.I)),
    (T.TECHNICAL_EXPLANATION, re.compile(r"\b(explain|how does|how do|why does|difference between|what is)\b.*\b(gradient|neural|network|transformer|algorithm|database|protocol|kernel|embedding|api|model)\b", re.I)),
]

# Words that start a new requested action; used to count subtasks.
_ACTION = re.compile(r"\b(explain|write|summari[sz]e|list|compare|describe|create|generate|give|implement|solve|analy[sz]e|translate|draft)\b", re.I)


def count_subtasks(prompt: str) -> int:
    numbered = len(re.findall(r"(?m)^\s*(?:\d+[\.\)]|[-*])\s+\S", prompt))
    if numbered >= 2:
        return numbered
    # split on commas / "and" / "then" / ";" and count segments that begin an action
    parts = re.split(r",|;|\band then\b|\bthen\b|\band\b", prompt, flags=re.I)
    actions = sum(1 for p in parts if _ACTION.match(p.strip()))
    return max(1, actions)


def classify_task(prompt: str) -> T:
    if count_subtasks(prompt) >= 2:
        return T.MULTI_PART
    for cat, pat in _RULES:
        if pat.search(prompt):
            return cat
    return T.GENERAL_QA
