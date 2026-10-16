from prompts import SECTION_NAMES, build_prompt, parse_sections


def test_build_prompt_has_all_sections():
    prompt = build_prompt("Analise.", "Cenário teste.")
    assert "### Instruction" in prompt
    assert "### Scenario" in prompt
    for section in SECTION_NAMES:
        assert f"{section}:" in prompt


def test_parse_sections():
    text = """Gargalo: Acoplamento.
Causa raiz: Integração ponto a ponto.
Padrão recomendado: Event-driven.
Solução: Kafka.
Justificativa técnica: Desacoplamento.
Tecnologias possíveis: Kafka.
Vantagens: Replay.
Limitações/Trade-offs: Operação."""
    parsed = parse_sections(text)
    assert parsed["Gargalo"] == "Acoplamento."
    assert parsed["Solução"] == "Kafka."
    assert parsed["Limitações/Trade-offs"] == "Operação."
