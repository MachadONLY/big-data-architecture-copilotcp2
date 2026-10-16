"""Prompt templates shared by training, evaluation and the local app."""

SECTION_NAMES = [
    "Gargalo",
    "Causa raiz",
    "Padrão recomendado",
    "Solução",
    "Justificativa técnica",
    "Tecnologias possíveis",
    "Vantagens",
    "Limitações/Trade-offs",
]

def build_prompt(instruction: str, scenario: str) -> str:
    """Build the canonical prompt used during SFT and inference."""
    return (
        "### Instruction\n"
        f"{instruction.strip()}\n\n"
        "### Scenario\n"
        f"{scenario.strip()}\n\n"
        "### Required answer format\n"
        "Gargalo:\n"
        "Causa raiz:\n"
        "Padrão recomendado:\n"
        "Solução:\n"
        "Justificativa técnica:\n"
        "Tecnologias possíveis:\n"
        "Vantagens:\n"
        "Limitações/Trade-offs:\n\n"
        "### Answer\n"
    )

def parse_sections(text: str) -> dict[str, str]:
    """Best-effort extraction of the expected structured sections."""
    result = {name: "" for name in SECTION_NAMES}
    current = None
    aliases = {
        "gargalo": "Gargalo",
        "causa raiz": "Causa raiz",
        "padrão recomendado": "Padrão recomendado",
        "padrao recomendado": "Padrão recomendado",
        "solução": "Solução",
        "solucao": "Solução",
        "justificativa técnica": "Justificativa técnica",
        "justificativa tecnica": "Justificativa técnica",
        "tecnologias possíveis": "Tecnologias possíveis",
        "tecnologias possiveis": "Tecnologias possíveis",
        "vantagens": "Vantagens",
        "limitações/trade-offs": "Limitações/Trade-offs",
        "limitacoes/trade-offs": "Limitações/Trade-offs",
        "trade-offs": "Limitações/Trade-offs",
        "limitações": "Limitações/Trade-offs",
        "limitacoes": "Limitações/Trade-offs",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = None
        lower = line.lower()
        for alias, canonical in aliases.items():
            if lower == alias or lower.startswith(alias + ":"):
                matched = canonical
                content = line.split(":", 1)[1].strip() if ":" in line else ""
                result[canonical] = content
                current = canonical
                break
        if matched:
            continue
        if current:
            result[current] = (result[current] + " " + line).strip()
    return result
