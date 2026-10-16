"""No-GPU UI illustrative mode: deterministic rules, explicitly not trained LLM."""
from offline_demo import illustrate, match_category
from prompts import parse_sections, SECTION_NAMES


def test_architecture_diagram_example():
    scenario = "Conectamos 8 produtores diretamente a 12 consumidores ponto a ponto. Adicionar um consumidor exige alterar vários sistemas."
    response = illustrate(scenario)
    assert response["category"] == "n_to_m_integrations"
    assert response["mode"] == "illustrative_rules"
    assert all(parse_sections(response["text"])[name] for name in SECTION_NAMES)


def test_oltp_conflict():
    assert match_category("Consultas analíticas pesadas bloqueiam meu banco transacional OLTP") == "oltp_vs_olap"


def test_unknown_does_not_pretend_diagnosis():
    answer = illustrate("O sistema não funciona.")
    assert answer["category"] is None
    assert "Não determinado" in answer["text"]


def test_empty_input_rejected():
    try:
        illustrate(" ")
    except ValueError:
        pass
    else:
        raise AssertionError("empty scenario should fail")
