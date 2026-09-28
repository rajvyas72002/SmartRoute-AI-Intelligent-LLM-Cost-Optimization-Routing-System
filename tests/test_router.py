from core.complexity_classifier import analyze
from core.router import route
from core.schemas import ModelSpec, RoutingSettings, Strategy, Complexity, TaskCategory
from config.settings import load_registry, load_routing_config

CFG = load_routing_config()
ALL = ["general_qa","summarization","rewriting","technical_explanation","coding","math","data_analysis","planning","multi_part"]

def mk(id, cap, tps=20, kind="local", pin=0.0, pout=0.0):
    return ModelSpec(id=id, provider="ollama", kind=kind, task_types=ALL, input_price_per_mtok=pin,
                     output_price_per_mtok=pout, context_window=4096, ttft_s=1, tokens_per_sec=tps, capability=cap)

REG = [mk("small", 2), mk("big", 3, tps=5), mk("cloud", 5, kind="cloud", tps=80, pin=1, pout=4)]
AV = {"small": True, "big": True, "cloud": True}

def test_simple_prompt_is_simple_and_qa():
    f = analyze("What is the capital of France?")
    assert f.complexity == Complexity.SIMPLE and f.category == TaskCategory.GENERAL_QA

def test_multipart_detected():
    f = analyze("Explain gradient descent, write a Python example, and summarize its advantages.")
    assert f.category == TaskCategory.MULTI_PART and f.subtask_count == 3

def test_math_proof_is_at_least_moderate():
    assert analyze("Prove step by step that sqrt(2) is irrational").complexity != Complexity.SIMPLE

def test_cost_strategy_picks_cheapest_capable_simple():
    d = route(analyze("What is the capital of France?"), REG, AV, RoutingSettings(strategy=Strategy.COST), CFG)
    assert d.chosen == "small"  # cost ties -> least over-qualified model

def test_cloud_blocked_unless_allowed():
    d = route(analyze("Prove step by step that sqrt(2) is irrational. Design a plan and explain, compare, justify."),
              REG, AV, RoutingSettings(strategy=Strategy.QUALITY, allow_cloud=False), CFG)
    assert d.chosen != "cloud"
    assert any(c.model_id == "cloud" and "disabled" in c.reason for c in d.candidates)

def test_quality_strategy_uses_cloud_when_allowed():
    d = route(analyze("Prove step by step that sqrt(2) is irrational"), REG, AV,
              RoutingSettings(strategy=Strategy.QUALITY, allow_cloud=True), CFG)
    assert d.chosen == "cloud"

def test_manual_override_respected_and_unavailable_is_error():
    f = analyze("hi")
    d = route(f, REG, AV, RoutingSettings(manual_model="small"), CFG)
    assert d.chosen == "small"
    d = route(f, REG, {"small": False}, RoutingSettings(manual_model="small"), CFG)
    assert d.chosen is None and d.error

def test_no_models_gives_clear_error():
    d = route(analyze("hi"), REG, {}, RoutingSettings(), CFG)
    assert d.chosen is None and "Ollama" in d.error

def test_capability_relaxation_is_disclosed():
    reg = [mk("small", 2)]
    d = route(analyze("Prove step by step that sqrt(2) is irrational. Design, compare, justify, analyze."),
              reg, {"small": True}, RoutingSettings(), CFG)
    assert d.chosen == "small" and d.constraints_violated

def test_yaml_registry_validates():
    models, _ = load_registry()
    assert len(models) >= 3
