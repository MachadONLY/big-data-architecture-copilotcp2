# Última auditoria independente — CP2/CP3 FIAP — 15/10/2026

## Status da entrega

**IMPLEMENTADO, MAS NÃO INTEGRALMENTE EXECUTADO/COMPROVADO.**

O código e a documentação podem ser inspecionados. Este pacote **não contém** adapters treinados, resultados de treinamento CUDA, avaliação Base vs Fine-tuned executada, screenshots de GPU/execução do frontend, ou evidência de publicação no GitHub. **O frontend não produzirá respostas do modelo fine-tuned sem um adapter treinado e dependências locais.** Foi incluído um modo ilustrativo baseado em regras, explicitamente identificado como NÃO sendo uma LLM, para inspecionar a interface sem GPU. Não declarar o trabalho 100% concluído.

## Requisitos do enunciado e evidências

| Requisito oficial | Situação comprovável neste ZIP |
|---|---|
| Escolher LLM | Modelo Qwen3-1.7B-Base identificado e justificado |
| Escolher problema específico | Diagnóstico de Big Data Architecture documentado |
| Testar diferentes parâmetros | 3 arquivos de configuração presentes; treinos não executados |
| Avaliar o modelo e respostas | Código e rubrica presentes; sem métricas medidas |
| Fazer os testes utilizando GPU | Não executado/comprovado |
| Criar frontend de prompts/respostas | Implementado; resposta real do modelo treinado não verificada |
| Print de GPU funcionando no VSCode | Ausente |
| Documentação e código no GitHub | Preparados localmente; publicação não comprovada |

A exigência de grupo de 4–5 pessoas também permanece pendente: este pacote registra apenas os 3 integrantes fornecidos.

## Últimos testes independentes (sem CUDA)

- Integridade do ZIP de origem (`unzip -t`): **PASS**.
- Manifesto SHA-256 de origem (`sha256sum -c`): **PASS**.
- `pytest` dos quatro módulos disponíveis sem a stack Hugging Face/Streamlit: **17 passed**.
- Duas verificações de API e renderização Streamlit: **não reexecutadas** no ambiente desta auditoria por ausência das bibliotecas necessárias; relatório anterior registra execução em outro ambiente.
- Compilação sintática Python (`compileall`): **PASS**.
- Auditoria offline `src/qa_static.py`: **PASS**.
- Validação `src/validate_dataset.py`: **PASS** (400 exemplos; 280/60/60; 20 categorias; sem duplicação exata de ID/input/prompt/output).
- Três YAMLs de experimentos: **PASS** em estrutura e controle de parâmetros.
- JSON e notebook: parse **PASS**.
- Resultados de treinamento: **0 linhas** preenchidas em `experiments/results.csv`.
- Adapters/pesos treinados: **nenhum**.
- Evidências de CUDA e screenshots reais: **nenhumas**.
- Scripts PowerShell: **revisão estática**, não execução Windows.

## Riscos para o professor executar

1. **Crítico/certo sem treinar:** o frontend não consegue responder com adapter fine-tuned porque o pacote não contém nenhum adapter.
2. **Crítico na rubrica:** a comprovação de GPU, três experimentos executados, benchmark e print obrigatório estão ausentes.
3. **Crítico na entrega:** o GitHub obrigatório ainda não está publicado/comprovado.
4. **Formal:** turma exige 4–5 integrantes; há 3 integrantes verificados.
5. **Compatibilidade:** a instalação do PyTorch CUDA deve ser escolhida com base na GPU/driver do PC. O ambiente final não foi testado.
6. **Metodologia:** dataset sintético apresenta semelhança elevada entre respostas de treino e teste; ausência de vazamento exato não comprova generalização semântica.

## Recomendação

Caso a entrega precise ser enviada sem GPU, envie este código como **entrega parcial verdadeira** e não como experimento executado. O enunciado requer explicitamente que o código/documentação estejam no GitHub; o ZIP isolado **não substitui** essa exigência. Para completar, seguir `ACTION_REQUIRED_TONIGHT.md`/`QUICKSTART.md` e gerar apenas evidências reais.
