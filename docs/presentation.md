# Roteiro de apresentação

## Slide 1 — Problema

**Big Data Architecture Copilot**

Modelos gerais conhecem conceitos de arquitetura, mas não foram especializados para diagnosticar nossos cenários seguindo um formato técnico consistente.

Mostrar um exemplo curto de cenário.

## Slide 2 — Hipótese

> Um Supervised Fine-Tuning especializado pode aumentar a consistência e a qualidade do diagnóstico arquitetural.

O objetivo não foi criar uma LLM do zero.

## Slide 3 — Modelo base

`Qwen/Qwen3-1.7B-Base`

Explicar:

- causal language model;
- 1.7B parâmetros;
- 28 camadas;
- contexto de 32.768 tokens;
- Apache 2.0;
- modelo Base escolhido para tornar o efeito do SFT mais observável;
- fallback: Qwen3-0.6B-Base se a VRAM real inviabilizar 1.7B.

## Slide 4 — Dataset

Mostrar:

- 400 exemplos;
- 20 categorias;
- 280 train;
- 60 validation;
- 60 test;
- test congelado.

Abrir um JSONL real.

Explicar que o dataset é sintético, versionado e regenerável.

## Slide 5 — Fine-tuning

Fluxo:

```text
Qwen3 Base
→ quantização 4-bit NF4
→ LoRA em all-linear
→ SFT
→ adapter especializado
```

Explicar que os pesos base ficam congelados e os adapters recebem atualização por gradiente.

## Slide 6 — GPU

Mostrar screenshot real do VSCode:

- treino executando;
- CUDA disponível;
- nome da GPU;
- `nvidia-smi`;
- `python.exe`;
- VRAM;
- GPU utilization.

Mostrar também `gpu_environment.json`.

## Slide 7 — Experimentos

Tabela:

| Exp | LR | Epochs | Rank | Alpha | Effective batch |
|---|---:|---:|---:|---:|---:|
| 01 | 2e-4 | 2 | 8 | 16 | 8 |
| 02 | 2e-4 | 2 | 16 | 32 | 8 |
| 03 | 1e-4 | 2 | 16 | 32 | 8 |

Explicar:

- 01 → 02: muda capacidade LoRA;
- 02 → 03: muda learning rate.

Depois mostrar os valores **reais** de loss, tempo e VRAM.

## Slide 8 — Avaliação

Comparação:

`BASE` vs `FINE-TUNED`

Mesmos 60 casos de teste, mesma estratégia de geração.

Métricas automáticas + rubrica humana cega.

Apresentar números reais.

## Slide 9 — Failure cases

Mostrar pelo menos um caso em que:

- fine-tuning melhorou;
- ambos falharam;
- fine-tuning piorou, se houver.

Isso demonstra rigor experimental.

## Slide 10 — Demo

Abrir Streamlit.

Digitar um cenário novo.

Mostrar:

- resposta estruturada;
- Running locally;
- modelo;
- adapter;
- tempo;
- GPU.

## Slide 11 — Conclusão

Responder somente com os dados reais:

1. a hipótese foi sustentada?
2. em quais métricas?
3. quais limitações permaneceram?
4. qual experimento teve melhor equilíbrio?
5. qual próximo passo seria justificável?

## Perguntas que o professor pode fazer

### “Isso é treinamento real?”

Sim. O backward atualiza parâmetros LoRA. Mostrar número de trainable parameters, loss e arquivos do adapter.

### “Por que não treinar tudo?”

Porque full fine-tuning de 1.7B exige muito mais memória e não é necessário para o objetivo. QLoRA reduz o custo mantendo adaptação real.

### “Por que Base?”

Para observar mais diretamente comportamento adquirido pelo SFT; o modelo Instruct já teria uma camada forte de instruction tuning.

### “Como sabem que não decorou?”

O test set não é carregado no treinamento. A avaliação usa cenários separados.

### “ROUGE prova que ficou melhor?”

Não. Por isso combinamos métricas automáticas com rubrica humana cega.

### “Onde está a GPU?”

Mostrar `nvidia-smi`, PyTorch CUDA e o processo consumindo VRAM durante o treino.
