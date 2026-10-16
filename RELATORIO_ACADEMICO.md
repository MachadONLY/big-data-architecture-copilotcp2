# Relatório técnico — Big Data Architecture Copilot

**FIAP | 4ESPV-2026 | CP2/CP3 — Generative AI for Engineering**  
**Professor:** Francisco Elanio Bezerra  
**Data de entrega:** 15/10/2026

**Integrantes informados:** Camila do Prado Padalino (RM98316), Gabriel Teixeira Machado (RM551570) e Felipe Cavalcante Bressane (RM97688). **Observação:** o enunciado prevê equipes de 4–5 integrantes; foi possível confirmar somente três nomes.

## Problema e objetivo

Equipes de engenharia de dados frequentemente descrevem sintomas arquiteturais sem um diagnóstico estruturado: integrações N×M, pressão no banco OLTP causada por consultas analíticas, consumer lag, falhas de retenção, CDC incorreto e problemas de consistência. A proposta é especializar uma LLM local para receber a descrição de um cenário e produzir: gargalo, causa raiz, padrão recomendado, solução, justificativa, tecnologias, vantagens e trade-offs.

## Escolha da LLM

Selecionamos `Qwen/Qwen3-1.7B-Base` por sua arquitetura causal, número moderado de parâmetros e disponibilidade sob licença Apache-2.0. O plano prevê fallback documentado para `Qwen/Qwen3-0.6B-Base`, condicionado à capacidade da GPU. O uso do modelo Base ajuda a examinar se o comportamento estruturado é adquirido por supervised fine-tuning em vez de ser inteiramente herdado de um modelo instruction-tuned. O motor de inferência implementado não depende de API comercial.

## Arquitetura e método

O pipeline versionado conecta: blueprints de cenários → geração reprodutível JSONL → divisão train/validation/test → tokenização prompt/completion → QLoRA/NF4 com PyTorch, Transformers, TRL e PEFT → adapters LoRA → seleção por validation loss → benchmark Base × Fine-tuned → frontend Streamlit. `src/train.py` exige CUDA e não carrega `test.jsonl`; `src/evaluate.py` foi projetado para avaliar o test set separado. Pesos grandes ficam fora do controle de versão.

## Dataset

Foram gerados **400 exemplos sintéticos** abrangendo **20 categorias**, divididos em **280 treino / 60 validação / 60 teste**, com semente fixa e arquivos separados. A validação executada confirmou quantidade, campos essenciais, balanceamento por categoria e ausência de duplicação textual exata ou de IDs entre os conjuntos. **Limitação:** os exemplos reutilizam estruturas de resposta e matrizes de contexto; sem revisão humana e sem benchmark executado, não é possível inferir boa generalização semântica apenas dessa separação.

## Experimentos planejados

| Experimento | Learning rate | Epochs | LoRA rank | Alpha | Effective batch | Objetivo |
|---|---:|---:|---:|---:|---:|---|
| 01 | 2e-4 | 2 | 8 | 16 | 8 | Referência de capacidade |
| 02 | 2e-4 | 2 | 16 | 32 | 8 | Maior capacidade do adapter |
| 03 | 1e-4 | 2 | 16 | 32 | 8 | Efeito da taxa de aprendizado |

Os demais valores estão registrados em `configs/*.yaml` e permanecem iguais entre os experimentos comparáveis. A análise A→B altera conjuntamente rank e alpha para manter a razão alpha/rank; B→C isola o learning rate. **Estas configurações foram implementadas, mas ainda não executadas neste pacote.**

## Avaliação prevista

O benchmark usa 60 cenários do test set. Serão avaliados termos esperados do gargalo, padrão arquitetural, cobertura de conceitos, conformidade do formato, ROUGE-L e afirmações tecnicamente proibidas. Essas métricas são heurísticas e **não equivalem a uma nota de correção técnica**. Uma rubrica humana adicional considera identificação, solução, justificativa, relevância, ausência de alucinação, completude e clareza, com pontuação cega A/B. Os scripts produzem CSV, JSON e casos em que o fine-tuning piorou. Não foram geradas métricas reais de Base × Fine-tuned porque o treinamento CUDA não foi realizado neste ambiente.

## Interface

O Streamlit em `app/app.py` possui entrada de prompt, resposta por seções e carregamento do modelo local + adapter PEFT quando disponível. Para permitir a inspeção da interface **sem baixar pesos ou dispor de CUDA**, foi incluído um **modo ilustrativo offline, baseado em regras**, que recebe cenários e mostra um exemplo estruturado. **Esse modo não usa uma LLM, não é fine-tuning e não conta como demonstração do requisito de GPU.** A UI exibe essa distinção ao usuário.

## Evidências e limitações

**Implementado e testado localmente:** integridade de arquivos, sintaxe, dataset, testes de unidade, validação das três configurações, geração de prompts, checagens do gate e demonstração baseada em regras. Consulte `QA_REPORT.md` e `VERIFICACAO_FINAL.md` para detalhes do ambiente desta auditoria.

**Não executado e não comprovado neste pacote:** treinamentos reais em GPU NVIDIA/CUDA, loss/duração/VRAM medidos, pesos LoRA treinados, benchmark executado, print da GPU no VSCode e inferência do adapter treinado. Também não há prova de publicação no GitHub.

## Como reproduzir

1. Instalar Python, Git e uma versão do PyTorch CUDA compatível com o driver NVIDIA do computador, conforme `docs/gpu.md` e `QUICKSTART.md`.
2. Instalar o restante do stack com `python -m pip install -r requirements.txt`.
3. Validar `python src/gpu_check.py --require-cuda` e `python src/verify_stack.py --require-cuda`.
4. Executar `scripts/run_pipeline.ps1` no Windows/PowerShell para treino, seleção e benchmark.
5. Salvar os screenshots reais de GPU e da inferência do adapter e versionar a documentação/resultados leves no GitHub.

**Referências:** model card oficial de `Qwen/Qwen3-1.7B-Base` (Hugging Face), documentação PEFT/LoRA, TRL/SFT, PyTorch/CUDA e bibliografia indicada em `docs/references.md`.
