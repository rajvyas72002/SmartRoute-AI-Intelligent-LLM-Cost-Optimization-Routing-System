import streamlit as st
from config.settings import load_registry, load_routing_config
from core.complexity_classifier import analyze
from core.router import route
from core.cost_calculator import cost_usd
from core.schemas import RoutingSettings, Strategy
from providers.provider_factory import get_provider

st.set_page_config(page_title="SmartRoute AI", page_icon="🧭", layout="wide")
registry, local_cost = load_registry()
cfg = load_routing_config()

@st.cache_data(ttl=15)
def availability_map() -> dict[str, bool]:
    return {m.id: get_provider(m.provider).is_available(m.id) for m in registry}

avail = availability_map()

with st.sidebar:
    st.title("🧭 SmartRoute AI")
    strategy = st.selectbox("Routing strategy", [s.value for s in Strategy], index=1)
    allow_cloud = st.toggle("Allow cloud models", value=False,
                            help="Prompts leave your machine when this is on.")
    manual = st.selectbox("Model", ["(automatic routing)"] + [m.id for m in registry])
    max_cost = st.number_input("Max cost / request (USD, 0 = no limit)", min_value=0.0, value=0.0, format="%.6f")
    max_lat = st.number_input("Max latency (s, 0 = no limit)", min_value=0.0, value=0.0)
    st.subheader("Model availability")
    for m in registry:
        st.write(("🟢" if avail[m.id] else "🔴"), m.id, f"({m.kind})")
    if st.button("Clear conversation"):
        st.session_state.history = []; st.rerun()

st.session_state.setdefault("history", [])
settings = RoutingSettings(strategy=Strategy(strategy), allow_cloud=allow_cloud,
                           manual_model=None if manual.startswith("(") else manual,
                           max_cost_usd=max_cost or None, max_latency_s=max_lat or None)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if "meta" in turn:
            st.caption(turn["meta"])
            with st.expander("Why this model?"):
                st.write(turn["why"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    features = analyze(prompt, cfg["complexity_thresholds"]["simple_max"], cfg["complexity_thresholds"]["moderate_max"])
    dec = route(features, registry, avail, settings, cfg, local_cost)
    with st.chat_message("assistant"):
        if dec.error:
            st.error(dec.error)
            st.session_state.history.append({"role": "assistant", "content": f"⚠️ {dec.error}"})
        else:
            spec = next(m for m in registry if m.id == dec.chosen)
            with st.spinner(f"Running {dec.chosen}..."):
                resp = get_provider(spec.provider).generate(dec.chosen, prompt)
            if resp.error:
                text = f"⚠️ Model error: {resp.error}"
            else:
                text = resp.text
            actual = cost_usd(spec, resp.input_tokens, resp.output_tokens, local_cost)
            label = "est. local cost" if spec.kind == "local" else "cost"
            meta = (f"**{dec.chosen}** · {features.category.value} · {features.complexity.value} · "
                    f"{resp.latency_s:.1f}s · {resp.input_tokens}+{resp.output_tokens} tokens · {label} ${actual:.6f}")
            why = dec.explanation + ("\n\n⚠️ " + " ".join(dec.constraints_violated) if dec.constraints_violated else "")
            st.markdown(text); st.caption(meta)
            with st.expander("Why this model?"):
                st.write(why)
                st.dataframe([c.model_dump() for c in dec.candidates], use_container_width=True)
            st.session_state.history.append({"role": "assistant", "content": text, "meta": meta, "why": why})
