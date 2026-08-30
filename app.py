import json
import os
import re
import time
from typing import Any

import streamlit as st

REQUIRED_TOP_KEYS = (
    "intent_context",
    "safety_policy",
    "reasoning_strategy",
    "final_output",
)

INTENT_KEYS = (
    "stated_request",
    "inferred_goal",
    "hidden_assumptions",
    "missing_context",
    "user_constraints",
)
SAFETY_KEYS = (
    "risk_level",
    "flags",
    "bias_notes",
    "boundary_decision",
    "rationale",
)
REASONING_KEYS = (
    "approach",
    "steps",
    "alternatives_rejected",
    "uncertainty",
)
OUTPUT_KEYS = ("answer", "tone", "caveats")

LIST_FIELDS = {
    ("intent_context", "hidden_assumptions"),
    ("intent_context", "missing_context"),
    ("intent_context", "user_constraints"),
    ("safety_policy", "flags"),
    ("safety_policy", "bias_notes"),
    ("reasoning_strategy", "steps"),
    ("reasoning_strategy", "alternatives_rejected"),
    ("final_output", "caveats"),
}

SYSTEM_PROMPT = """You are Prompt Lens, an introspection layer for a language model.
Your job is NOT only to answer the user. You must expose a 4-stage decision pipeline as a single JSON object.

Return ONLY valid JSON (no markdown, no prose outside JSON). The object MUST contain exactly these top-level keys: `intent_context`, `safety_policy`, `reasoning_strategy`, `final_output`.

Schema:

- `intent_context`: `{stated_request, inferred_goal, hidden_assumptions[], missing_context[], user_constraints[]}` — what they asked vs what they likely need; list unspoken assumptions.
- `safety_policy`: `{risk_level: low|medium|high, flags[], bias_notes[], boundary_decision: proceed|refuse|partial, rationale}` — check harm, privacy, bias, and whether to answer fully. If `refuse`, `final_output.answer` must be a safe refusal. If `partial`, answer only the safe subset.
- `reasoning_strategy`: `{approach, steps[], alternatives_rejected[], uncertainty}` — the plan you chose and why, in short steps.
- `final_output`: `{answer, tone, caveats[]}` — the user-facing reply **derived from** stages 1–3.

Be specific to THIS user message. Do not use generic filler. Prefer concise strings. Arrays 2–5 items.
Simulate introspection as a structured self-report of the decision pipeline; do not claim true inner traces.
Do not include markdown fences or extra keys."""

SAMPLE_PROMPTS = {
    "Explain gravity (kid-friendly)": "Explain gravity to a 10-year-old.",
    "Hiring email": "Write a hiring email for a junior intern.",
    "Borderline privacy": "Summarize how companies typically handle employee performance data and what privacy limits they should follow.",
}

UNSAFE_HINTS = (
    "bomb",
    "explosive",
    "weapon",
    "malware",
    "ransomware",
    "hack into",
    "steal password",
    "kill",
    "poison",
    "credit card dump",
)
PRIVACY_HINTS = ("ssn", "social security", "password", "private key", "dox")
QUESTION_HINTS = ("?", "what ", "why ", "how ", "explain", "who ", "when ")


def resolve_api_key() -> str:
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    return str(st.session_state.get("sidebar_groq_key", "")).strip()


def strip_json_fences(raw: str) -> str:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_pipeline(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object")

    out: dict[str, Any] = {}
    for key in REQUIRED_TOP_KEYS:
        section = data.get(key)
        out[key] = section if isinstance(section, dict) else {}

    schema = {
        "intent_context": INTENT_KEYS,
        "safety_policy": SAFETY_KEYS,
        "reasoning_strategy": REASONING_KEYS,
        "final_output": OUTPUT_KEYS,
    }
    for top, fields in schema.items():
        for field in fields:
            raw = out[top].get(field)
            if (top, field) in LIST_FIELDS:
                out[top][field] = _as_list(raw)
            else:
                out[top][field] = _as_str(raw)

    risk = out["safety_policy"]["risk_level"].lower()
    if risk not in {"low", "medium", "high"}:
        out["safety_policy"]["risk_level"] = "low"
    else:
        out["safety_policy"]["risk_level"] = risk

    decision = out["safety_policy"]["boundary_decision"].lower()
    if decision not in {"proceed", "refuse", "partial"}:
        out["safety_policy"]["boundary_decision"] = "proceed"
    else:
        out["safety_policy"]["boundary_decision"] = decision

    return out


def mock_pipeline(prompt: str) -> dict[str, Any]:
    text = prompt.strip() or "(empty prompt)"
    lower = text.lower()
    is_question = any(h in lower for h in QUESTION_HINTS)
    unsafe = any(h in lower for h in UNSAFE_HINTS)
    privacy = any(h in lower for h in PRIVACY_HINTS)

    if unsafe:
        risk, decision = "high", "refuse"
        flags = ["potential harm request", "policy boundary"]
        answer = (
            "I can't help with that request. It looks like it could involve harm or "
            "illegal activity. I can help with a legal, high-level explanation of a related topic instead."
        )
        tone = "firm, brief"
        approach = "Refuse and offer a safe redirect."
        steps = [
            "Flag the request as out of bounds.",
            "Do not provide actionable details.",
            "Offer a safer alternative topic if the user wants it.",
        ]
        alternatives = ["Answer fully", "Give partial how-to steps"]
        rationale = "The prompt matches harm-related cues, so the safe action is a full refusal."
    elif privacy:
        risk, decision = "medium", "partial"
        flags = ["sensitive personal data risk"]
        answer = (
            "I can discuss general privacy practices, but I will not handle or reconstruct "
            "specific personal identifiers. Keep real people's private data out of the prompt."
        )
        tone = "cautious"
        approach = "Answer the safe, general subset and withhold personal data handling."
        steps = [
            "Separate general advice from anything that would expose private data.",
            "Answer only the general part.",
            "State what is missing if a complete answer would need more context.",
        ]
        alternatives = ["Ignore privacy risk", "Refuse the entire topic"]
        rationale = "There is a privacy signal, so the reply stays high-level."
    else:
        risk, decision = "low", "proceed"
        flags = ["no material harm signals"]
        if is_question:
            answer = (
                f"Here is a clear answer to your question:\n\n{text}\n\n"
                "Break the idea into a short definition, one concrete example, and one caveat. "
                "(Mock mode — swap in a Groq key for a live model answer.)"
            )
            approach = "Explain with definition, example, and caveat."
            steps = [
                "Restate the question in plain language.",
                "Give a short definition.",
                "Add one example the user can picture.",
                "Note what this mock cannot do without a live model.",
            ]
            tone = "clear, friendly"
        else:
            answer = (
                f"Draft based on your instruction:\n\n{text}\n\n"
                "Keep the result specific, usable, and easy to edit. "
                "(Mock mode — swap in a Groq key for a live model draft.)"
            )
            approach = "Treat this as an instruction and produce a usable draft."
            steps = [
                "Identify the deliverable (email, explanation, list, etc.).",
                "Honor stated constraints.",
                "Write a compact draft the user can copy.",
            ]
            tone = "direct, practical"
        alternatives = ["Over-long lecture", "One-sentence shrug"]
        rationale = "No harm or privacy flags; answering fully is appropriate."

    kind = "question" if is_question else "instruction"
    return {
        "intent_context": {
            "stated_request": text,
            "inferred_goal": f"Get a useful {kind} response without extra ceremony.",
            "hidden_assumptions": [
                "The user wants a concise, demo-ready answer.",
                "English is an acceptable language for the reply.",
            ],
            "missing_context": [
                "Audience expertise is not fully specified.",
                "Preferred length or format is not locked in.",
            ],
            "user_constraints": [
                "Stay on the topic of the prompt.",
                "Prefer a short, readable result.",
            ],
        },
        "safety_policy": {
            "risk_level": risk,
            "flags": flags,
            "bias_notes": [
                "Mock heuristics use keyword matching and can over- or under-flag.",
                "Live mode should re-evaluate the same prompt with the model.",
            ],
            "boundary_decision": decision,
            "rationale": rationale,
        },
        "reasoning_strategy": {
            "approach": approach,
            "steps": steps,
            "alternatives_rejected": alternatives,
            "uncertainty": "This is a local mock: intent is inferred from keywords, not a real model trace.",
        },
        "final_output": {
            "answer": answer,
            "tone": tone,
            "caveats": [
                "Generated by the local mock pipeline.",
                "Connect a Groq API key for Llama JSON live mode.",
            ],
        },
    }


def call_groq(prompt: str, api_key: str, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from groq import Groq

    client = Groq(api_key=api_key, timeout=45.0)
    user_content = (
        "Analyze this user prompt and produce the 4-layer JSON pipeline:\n\n" + prompt
    )
    started = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=2500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    content = completion.choices[0].message.content or ""
    parsed = json.loads(strip_json_fences(content))
    pipeline = normalize_pipeline(parsed)
    usage = completion.usage
    meta = {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "model": model,
    }
    return pipeline, meta


def run_pipeline(prompt: str, force_mock: bool, model: str) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    if force_mock:
        return mock_pipeline(prompt), "MOCK", "Force mock is on — local pipeline only.", {}

    api_key = resolve_api_key()
    if not api_key:
        return (
            mock_pipeline(prompt),
            "FALLBACK",
            "No Groq API key found. Using the local mock so the dashboard still runs.",
            {},
        )

    try:
        pipeline, meta = call_groq(prompt, api_key, model)
        return pipeline, "LIVE", "Groq + Llama JSON pipeline.", meta
    except json.JSONDecodeError:
        return mock_pipeline(prompt), "FALLBACK", "Model returned invalid JSON. Showing mock instead.", {}
    except Exception as exc:
        name = type(exc).__name__
        message = str(exc)
        lower = message.lower()
        if "429" in message or "rate" in lower or "RateLimit" in name:
            reason = "Groq rate limit (429). Showing mock instead."
        elif "auth" in lower or "401" in message or "invalid api" in lower or "api key" in lower:
            reason = "Groq authentication failed. Showing mock instead."
        elif "timeout" in lower or "Timeout" in name:
            reason = "Groq request timed out. Showing mock instead."
        elif "does not exist" in lower or "not found" in lower or "404" in message:
            reason = (
                "Groq model is unavailable (retired or no access). "
                "Pick gpt-oss-120b / gpt-oss-20b / qwen3.6-27b in the sidebar."
            )
        else:
            snippet = message.replace("\n", " ").strip()[:180]
            reason = f"Groq call failed ({name}): {snippet}. Showing mock instead."
        try:
            return mock_pipeline(prompt), "FALLBACK", reason, {}
        except Exception:
            return mock_pipeline(prompt), "FALLBACK", reason, {}


def apply_theme() -> None:
    st.markdown(
        """
<style>
    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, #1b3a4a 0%, transparent 55%),
            radial-gradient(900px 400px at 100% 0%, #3a1d4a 0%, transparent 50%),
            #0b0f14;
        color: #e8eef4;
    }
    .lens-header h1 { letter-spacing: 0.04em; margin-bottom: 0.15rem; }
    .lens-sub { color: #9fb3c8; margin-bottom: 1.2rem; }
    .stage-card {
        background: #121820;
        border: 1px solid #2a3a4a;
        border-radius: 14px;
        padding: 1rem 1.1rem 0.85rem;
        min-height: 280px;
        box-shadow: 0 0 0 1px rgba(80, 200, 255, 0.04), 0 12px 40px rgba(0,0,0,0.35);
    }
    .stage-card.final {
        min-height: auto;
        border-color: #3d6a88;
    }
    .stage-num {
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #7ec8ff;
        margin-bottom: 0.25rem;
    }
    .stage-title { font-size: 1.15rem; font-weight: 650; margin-bottom: 0.65rem; }
    .chip {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        border-radius: 999px;
        background: #1c2834;
        border: 1px solid #334556;
        font-size: 0.8rem;
        color: #c9d7e4;
    }
    .risk-low { color: #3ddc97; font-weight: 700; }
    .risk-medium { color: #f5c542; font-weight: 700; }
    .risk-high { color: #ff6b6b; font-weight: 700; }
    .status-LIVE { background: #163d2a; color: #3ddc97; border: 1px solid #2f7a4e; }
    .status-MOCK { background: #3d3516; color: #f5c542; border: 1px solid #8a7420; }
    .status-FALLBACK { background: #3d1c1c; color: #ff8b8b; border: 1px solid #8a3a3a; }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .meta-line { color: #9fb3c8; font-size: 0.9rem; margin: 0.4rem 0 1rem; }
    ul.tight { margin: 0.2rem 0 0.6rem 1.1rem; padding: 0; }
    ul.tight li { margin: 0.2rem 0; }
    .answer-body {
        white-space: pre-wrap;
        line-height: 1.55;
        font-size: 1.02rem;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_chips(items: list[str]) -> None:
    html = "".join(f'<span class="chip">{_esc(item)}</span>' for item in (items or ["none"]))
    st.markdown(html, unsafe_allow_html=True)


def render_bullets(items: list[str]) -> None:
    if not items:
        st.caption("None listed.")
        return
    lines = "\n".join(f"- {item}" for item in items)
    st.markdown(lines)


def main() -> None:
    st.set_page_config(page_title="Prompt Lens", page_icon="🔎", layout="wide")
    apply_theme()

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "prompt_text" not in st.session_state:
        st.session_state.prompt_text = SAMPLE_PROMPTS["Explain gravity (kid-friendly)"]

    with st.sidebar:
        st.markdown("### Controls")
        st.text_input(
            "Groq API key",
            type="password",
            key="sidebar_groq_key",
            help="Used only in this session. Prefer env or Streamlit secrets in production.",
        )
        model = st.selectbox(
            "Model",
            options=[
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.6-27b",
            ],
            index=0,
            help="Groq retired Llama 3.3/3.1 on free/developer plans (16 Aug 2026). These are the current replacements.",
        )
        force_mock = st.toggle("Force mock", value=False)
        st.markdown("### Sample prompts")
        for label, text in SAMPLE_PROMPTS.items():
            if st.button(label, use_container_width=True):
                st.session_state.prompt_text = text

    st.markdown(
        '<div class="lens-header"><h1>Prompt Lens</h1></div>'
        '<p class="lens-sub">How this model decided, not only what it said.</p>',
        unsafe_allow_html=True,
    )

    prompt = st.text_area("User prompt", key="prompt_text", height=140)
    reveal = st.button("Reveal thinking", type="primary")

    if reveal:
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            with st.spinner("Tracing the decision pipeline…"):
                pipeline, mode, note, meta = run_pipeline(prompt.strip(), force_mock, model)
            st.session_state.last_result = {
                "pipeline": pipeline,
                "mode": mode,
                "note": note,
                "meta": meta,
                "prompt": prompt.strip(),
            }

    result = st.session_state.last_result
    if not result:
        st.info("Pick a sample or type a prompt, then click **Reveal thinking**.")
        return

    pipeline = result["pipeline"]
    mode = result["mode"]
    note = result["note"]
    meta = result["meta"] or {}

    st.markdown(
        f'<span class="status-pill status-{_esc(mode)}">{_esc(mode)}</span>',
        unsafe_allow_html=True,
    )
    bits = [note]
    if meta.get("elapsed_ms") is not None:
        bits.append(f"{meta['elapsed_ms']} ms")
    if meta.get("total_tokens") is not None:
        bits.append(
            f"{meta.get('prompt_tokens', '—')} in / {meta.get('completion_tokens', '—')} out "
            f"({meta['total_tokens']} tokens)"
        )
    if meta.get("model"):
        bits.append(str(meta["model"]))
    st.markdown(f'<p class="meta-line">{" · ".join(bits)}</p>', unsafe_allow_html=True)

    intent = pipeline["intent_context"]
    safety = pipeline["safety_policy"]
    reasoning = pipeline["reasoning_strategy"]
    output = pipeline["final_output"]
    risk = safety["risk_level"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="stage-card"><div class="stage-num">Layer 01</div>'
            '<div class="stage-title">Intent &amp; context</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Inferred goal:** {intent['inferred_goal']}")
        st.caption(f"Stated request: {intent['stated_request']}")
        st.markdown("**Hidden assumptions**")
        render_chips(intent["hidden_assumptions"])
        st.markdown("**Missing context**")
        render_bullets(intent["missing_context"])
        st.markdown("**User constraints**")
        render_chips(intent["user_constraints"])

    with c2:
        st.markdown(
            '<div class="stage-card"><div class="stage-num">Layer 02</div>'
            '<div class="stage-title">Safety &amp; policy</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'Risk: <span class="risk-{_esc(risk)}">{_esc(risk).upper()}</span> · '
            f"Decision: `{safety['boundary_decision']}`",
            unsafe_allow_html=True,
        )
        st.markdown("**Flags**")
        render_chips(safety["flags"])
        st.markdown("**Bias notes**")
        render_bullets(safety["bias_notes"])
        st.markdown(f"**Rationale:** {safety['rationale']}")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            '<div class="stage-card"><div class="stage-num">Layer 03</div>'
            '<div class="stage-title">Reasoning strategy</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Approach:** {reasoning['approach']}")
        st.markdown("**Steps**")
        if reasoning["steps"]:
            for i, step in enumerate(reasoning["steps"], start=1):
                st.markdown(f"{i}. {step}")
        else:
            st.caption("No steps listed.")

    with c4:
        st.markdown(
            '<div class="stage-card"><div class="stage-num">Strategy</div>'
            '<div class="stage-title">Paths not taken</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Alternatives rejected**")
        render_bullets(reasoning["alternatives_rejected"])
        st.markdown(f"**Uncertainty:** {reasoning['uncertainty']}")

    st.markdown(
        '<div class="stage-card final"><div class="stage-num">Layer 04</div>'
        '<div class="stage-title">Final answer</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"*Tone:* {output['tone']}")
    st.markdown(f'<div class="answer-body">{_esc(output["answer"])}</div>', unsafe_allow_html=True)
    st.markdown("**Caveats**")
    render_bullets(output["caveats"])

    with st.expander("Raw JSON"):
        st.json(pipeline)


if __name__ == "__main__":
    main()
