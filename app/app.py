from __future__ import annotations

import os
import json
import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompts import SECTION_NAMES, parse_sections  # noqa: E402
from offline_demo import illustrate  # noqa: E402

st.set_page_config(
    page_title="Big Data Architecture Copilot",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --bg: #0a0d12;
  --panel: #11161f;
  --panel-2: #151c27;
  --line: rgba(255,255,255,.08);
  --muted: #8f9bad;
  --text: #f6f8fb;
  --accent: #71e0c0;
}
.stApp { background: radial-gradient(circle at 20% 0%, #172033 0, #0a0d12 38%, #090b0f 100%); }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -0.035em; }
.hero {
  border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(255,255,255,.055), rgba(255,255,255,.018));
  border-radius: 24px;
  padding: 28px 30px;
  margin-bottom: 22px;
  box-shadow: 0 22px 70px rgba(0,0,0,.28);
}
.eyebrow { color: #71e0c0; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
.hero-title { color: var(--text); font-size: 2.35rem; font-weight: 750; margin: 8px 0 8px 0; }
.hero-copy { color: #aab4c3; max-width: 780px; line-height: 1.55; }
.status-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
.pill {
  border:1px solid var(--line); border-radius:999px; padding:7px 11px;
  background:rgba(255,255,255,.035); color:#cad2de; font-size:.78rem;
}
.dot { display:inline-block;width:8px;height:8px;border-radius:50%;background:#71e0c0;margin-right:7px;box-shadow:0 0 12px #71e0c0; }
.answer-card {
  border:1px solid var(--line); border-radius:18px; padding:18px 20px;
  background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
  margin:10px 0;
}
.answer-label { color:#8f9bad; font-size:.72rem; letter-spacing:.11em; text-transform:uppercase; font-weight:700; }
.answer-text { color:#edf1f7; line-height:1.58; margin-top:7px; }
.meta { color:#8f9bad; font-size:.8rem; }
div[data-testid="stTextArea"] textarea {
  min-height: 190px; border-radius: 16px !important; background:#0e131b !important;
}
div[data-testid="stButton"] button {
  border-radius: 13px; height: 3rem; font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


def adapters() -> list[Path]:
    env = os.getenv("COPILOT_ADAPTER")
    found: list[Path] = []
    if env and Path(env).exists():
        found.append(Path(env))

    pointer = ROOT / "experiments" / "best_adapter.txt"
    if pointer.exists():
        candidate = ROOT / pointer.read_text(encoding="utf-8").strip()
        if candidate.exists() and candidate not in found:
            found.append(candidate)

    for path in sorted((ROOT / "experiments").glob("experiment_*/adapter")):
        if path.exists() and path not in found:
            found.append(path)
    return found


@st.cache_resource(show_spinner="Carregando modelo local na GPU...")
def load_copilot(model_name: str, adapter: str):
    # Import the GPU-dependent stack only when a trained adapter is selected.
    from inference import LocalArchitectureCopilot
    return LocalArchitectureCopilot(model_name=model_name, adapter_path=adapter)


available = adapters()
def resolve_base_model(adapter: str | None) -> str:
    if not adapter:
        return "Qwen/Qwen3-1.7B-Base"
    config = Path(adapter) / "adapter_config.json"
    if config.exists():
        try:
            recorded = json.loads(config.read_text(encoding="utf-8")).get("base_model_name_or_path")
            if recorded:
                return str(recorded)
        except (OSError, json.JSONDecodeError):
            pass
    return "Qwen/Qwen3-1.7B-Base"

model_name = "Qwen/Qwen3-1.7B-Base"

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">FIAP · 4ESPV-2026 · Local AI</div>
  <div class="hero-title">Big Data Architecture Copilot</div>
  <div class="hero-copy">
    LLM especializada em diagnosticar gargalos de arquitetura de dados e recomendar
    padrões, tecnologias e trade-offs. Inferência executada localmente, sem API comercial.
  </div>
  <div class="status-row">
    <span class="pill">Local inference · disponível após treinamento</span>
    <span class="pill">QLoRA / PEFT</span>
    <span class="pill">GPU CUDA necessária</span>
    <span class="pill">Qwen3 Base</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Runtime")
    st.caption("O frontend carrega o modelo base + adapter LoRA diretamente na máquina.")
    if available:
        selected = st.selectbox(
            "Adapter",
            options=[str(p) for p in available],
            format_func=lambda x: str(Path(x).relative_to(ROOT)) if Path(x).is_relative_to(ROOT) else x,
        )
    else:
        selected = None
        st.info("Nenhum adapter treinado foi encontrado. Modo de ilustração sem LLM disponível para inspeção da interface.")
    model_name = resolve_base_model(selected)
    st.text_input("Modelo base", value=model_name, disabled=True)
    st.divider()
    st.caption("Nenhuma chamada OpenAI, Claude ou Gemini é feita por este app.")
    if not available:
        st.caption("O modo de ilustração não usa LLM, GPU nem produz evidências experimentais.")

col1, col2 = st.columns([1.15, 0.85], gap="large")

with col1:
    st.subheader("Descreva seu cenário")
    scenario = st.text_area(
        "Cenário de arquitetura",
        label_visibility="collapsed",
        placeholder=(
            "Ex.: Temos oito serviços produtores e doze consumidores conectados diretamente. "
            "Adicionar um consumidor exige alterar vários produtores e hoje não conseguimos "
            "reprocessar eventos históricos..."
        ),
    )
    analyze = st.button("Analisar arquitetura", type="primary", use_container_width=True)

with col2:
    st.subheader("O que o modelo avalia")
    st.markdown(
        """
- gargalo arquitetural;
- causa raiz;
- padrão recomendado;
- solução proposta;
- justificativa técnica;
- tecnologias possíveis;
- vantagens;
- limitações e trade-offs.
"""
    )

if analyze:
    if not scenario.strip():
        st.warning("Descreva um cenário antes de analisar.")
        st.stop()

    try:
        if selected:
            copilot = load_copilot(model_name, selected)
            with st.spinner("Executando inferência local..."):
                result = copilot.analyze(scenario.strip())
        else:
            example = illustrate(scenario.strip())
            result = {"text": example["text"], "sections": parse_sections(example["text"]), "mode": "illustrative_rules"}
            st.warning("DEMONSTRAÇÃO ILUSTRATIVA — baseada em regras, SEM LLM ou inferência CUDA. Não é o resultado do fine-tuning exigido pelo professor.")
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.divider()
    st.subheader("Diagnóstico")
    sections = result["sections"]
    for name in SECTION_NAMES:
        value = sections.get(name, "").strip()
        if value:
            safe_name = html.escape(name)
            safe_value = html.escape(value).replace("\n", "<br>")
            st.markdown(
                f'<div class="answer-card"><div class="answer-label">{safe_name}</div>'
                f'<div class="answer-text">{safe_value}</div></div>',
                unsafe_allow_html=True,
            )

    if not any(sections.values()):
        st.markdown(result["text"])

    if selected:
        st.caption(f"Inferência local · {result['elapsed_seconds']:.2f}s · {result['gpu']} · {result['precision']} · {Path(selected).name}")
    else:
        st.caption("Resposta de exemplo baseada em regras · nenhuma LLM carregada · nenhuma GPU utilizada")

    with st.expander("Resposta bruta do modelo"):
        st.code(result["text"], language=None)
