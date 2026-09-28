"""Routing engine: filter -> score -> explain. Knows nothing about providers or the UI."""
from core.schemas import (Candidate, ModelSpec, RoutingDecision, RoutingSettings, Strategy, TaskFeatures)
from core.cost_calculator import cost_usd, latency_s


def route(features: TaskFeatures, registry: list[ModelSpec], availability: dict[str, bool],
          settings: RoutingSettings, cfg: dict, local_cost_per_mtok: float = 0.0) -> RoutingDecision:
    out_tok = cfg["expected_output_tokens"][features.complexity.value]
    need_cap = cfg["required_capability"][features.complexity.value]
    dec = RoutingDecision(features=features, strategy=settings.strategy, chosen=None)

    def est(m):  # (cost, latency)
        return (cost_usd(m, features.est_input_tokens, out_tok, local_cost_per_mtok), latency_s(m, out_tok))

    # ---- manual override: respect it, never silently swap -------------------
    if settings.manual_model:
        m = next((x for x in registry if x.id == settings.manual_model), None)
        if m is None:
            dec.error = f"Manual model '{settings.manual_model}' is not in the registry."
        elif not availability.get(m.id, False):
            dec.error = f"Manual model '{m.id}' is not available (is it pulled in Ollama / is the API key set?)."
        else:
            c, l = est(m)
            dec.chosen, dec.est_cost_usd, dec.est_latency_s = m.id, c, l
            dec.explanation = f"Manual override: you selected {m.id}. Routing was skipped."
            if m.capability < need_cap:
                dec.constraints_violated.append(f"{m.id} capability {m.capability} < recommended {need_cap} for a {features.complexity.value} task")
        return dec

    # ---- Step 1: hard filters, recording WHY each model is in or out --------
    def check(m: ModelSpec, ignore_capability=False, ignore_limits=False) -> tuple[bool, str]:
        c, l = est(m)
        if not m.enabled: return False, "disabled in registry"
        if not availability.get(m.id, False): return False, "not available"
        if m.kind == "cloud" and not settings.allow_cloud: return False, "cloud providers disabled"
        if features.category.value not in m.task_types: return False, f"does not list task type '{features.category.value}'"
        if features.est_input_tokens + out_tok > m.context_window: return False, "context window too small"
        if not ignore_capability and m.capability < need_cap: return False, f"capability {m.capability} < required {need_cap}"
        if not ignore_limits:
            if settings.max_cost_usd is not None and c > settings.max_cost_usd: return False, f"est. cost ${c:.6f} > limit"
            if settings.max_latency_s is not None and l > settings.max_latency_s: return False, f"est. latency {l:.1f}s > limit"
        return True, "eligible"

    def build(**kw):
        return [Candidate(model_id=m.id, eligible=(r := check(m, **kw))[0], reason=r[1],
                          est_cost_usd=est(m)[0], est_latency_s=est(m)[1]) for m in registry]

    cands = build()
    pool = [c for c in cands if c.eligible]

    # ---- graceful relaxation (always disclosed) -----------------------------
    if not pool:
        relaxed = build(ignore_capability=True)
        pool = [c for c in relaxed if c.eligible]
        if pool:
            dec.constraints_violated.append(f"No model met required capability {need_cap}; using best available.")
            cands = relaxed
    if not pool:
        relaxed = build(ignore_capability=True, ignore_limits=True)
        pool = [c for c in relaxed if c.eligible]
        if pool:
            dec.constraints_violated.append("No model met your cost/latency limits; limits were exceeded.")
            cands = relaxed
    dec.candidates = cands
    if not pool:
        dec.error = "No available model can handle this request. Start Ollama and pull a model (see README)."
        return dec

    # ---- Step 2: strategy scoring -------------------------------------------
    cap = {m.id: m.capability for m in registry}
    if settings.strategy == Strategy.COST:
        key = lambda c: (round(c.est_cost_usd, 9), cap[c.model_id], c.est_latency_s)
        best = min(pool, key=key)
        why = "lowest estimated cost among eligible models (ties -> least over-qualified, then fastest)"
    elif settings.strategy == Strategy.QUALITY:
        best = max(pool, key=lambda c: (cap[c.model_id], -c.est_cost_usd))
        why = "highest capability among eligible models (ties -> cheaper)"
    else:
        w = cfg["balanced_weights"]
        mc, ml = max(c.est_cost_usd for c in pool) or 1.0, max(c.est_latency_s for c in pool) or 1.0
        score = lambda c: w["quality"] * cap[c.model_id] / 5 - w["cost"] * c.est_cost_usd / mc - w["latency"] * c.est_latency_s / ml
        best = max(pool, key=score)
        why = f"best weighted score (quality {w['quality']}, cost {w['cost']}, latency {w['latency']})"

    dec.chosen, dec.est_cost_usd, dec.est_latency_s = best.model_id, best.est_cost_usd, best.est_latency_s
    others = [c.model_id for c in pool if c.model_id != best.model_id]
    dec.explanation = (f"Detected '{features.category.value}' / {features.complexity.value} (score {features.score}: "
                       f"{'; '.join(features.reasons)}). Strategy '{settings.strategy.value}' chose {best.model_id}: {why}. "
                       f"Other eligible: {', '.join(others) or 'none'}.")
    return dec
