"""Lightweight dependency/API and frontend checks; no model download or CUDA needed."""
import pytest

from verify_stack import EXPECTED, collect


def test_pinned_qlora_api_contracts():
    pytest.importorskip("trl", reason="Full ML stack is not installed in offline audit environment")
    pytest.importorskip("bitsandbytes")
    result = collect(require_cuda=False)
    assert result["ready"], result
    assert result["expected_versions"] == EXPECTED
    assert result["api_checks"]["sft_config"] is True
    assert result["api_checks"]["bitsandbytes_config"] is True


def test_streamlit_frontend_renders_without_loading_a_model():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    app = AppTest.from_file("app/app.py").run(timeout=20)
    assert not app.exception
    assert len(app.text_area) == 1
    assert len(app.button) == 1
    assert app.button[0].label == "Analisar arquitetura"
    # A reviewer without CUDA can still inspect a clearly labelled rules-only demo.
    if "Nenhum adapter treinado" in str(app.info):
        app.text_area[0].set_value("Sistemas ponto a ponto: adicionar um consumidor exige alterar todos os produtores.").run()
        app.button[0].click().run()
        assert not app.exception
        assert app.warning  # Explicit disclaimer: not a trained LLM response.
