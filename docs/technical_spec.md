# Especificação técnica curta

## Nome

**Big Data Architecture Copilot**

## Objetivo

Especializar uma LLM local para diagnosticar cenários de arquitetura de dados e produzir respostas estruturadas com:

1. gargalo;
2. causa raiz;
3. padrão recomendado;
4. solução;
5. justificativa técnica;
6. tecnologias possíveis;
7. vantagens;
8. limitações/trade-offs.

## Modelo

Primário: `Qwen/Qwen3-1.7B-Base`.

Fallback condicionado à VRAM real: `Qwen/Qwen3-0.6B-Base`.

## Treinamento

- Supervised Fine-Tuning;
- QLoRA;
- quantização 4-bit NF4;
- double quantization;
- LoRA `all-linear`;
- CUDA obrigatório para experimentos oficiais;
- completion-only loss;
- BF16 quando suportado, senão FP16.

## Dataset

- 400 exemplos;
- 20 categorias;
- train: 280;
- validation: 60;
- test: 60;
- test set não utilizado durante treino.

## Experimentos

Três experimentos controlados:

- E1: rank 8 / alpha 16 / LR 2e-4;
- E2: rank 16 / alpha 32 / LR 2e-4;
- E3: rank 16 / alpha 32 / LR 1e-4.

Demais parâmetros principais constantes.

## Avaliação

Comparação do mesmo modelo Base contra o modelo Base + adapter fine-tuned no mesmo test set.

Métricas automáticas:

- bottleneck term hit;
- pattern term hit;
- concept coverage;
- format compliance;
- forbidden-claim hit;
- ROUGE-L.

Avaliação humana cega 0–4 em sete dimensões.

## Frontend

Streamlit local.

Entrada: cenário de arquitetura.

Saída: oito seções estruturadas + modelo + adapter + GPU + tempo de inferência.

## Evidência

- `gpu_check.py`;
- `verify_stack.py` com smoke NF4 em CUDA;
- `nvidia-smi`;
- processo Python consumindo VRAM;
- losses reais;
- duração real;
- peak VRAM;
- adapters;
- benchmark;
- screenshots;
- GitHub.

## Regra de integridade

Nenhuma métrica de treinamento, GPU ou resultado experimental é preenchida antes da execução real.
