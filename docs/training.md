# Treinamento

## Método

Supervised Fine-Tuning (SFT) com QLoRA:

1. carrega `Qwen/Qwen3-1.7B-Base`;
2. quantiza os pesos base em 4-bit NF4;
3. prepara o modelo para treino k-bit;
4. adiciona adapters LoRA em todas as camadas lineares;
5. calcula loss apenas sobre a `completion`;
6. treina usando CUDA;
7. valida em `validation.jsonl`;
8. salva somente adapter + tokenizer;
9. registra ambiente e métricas reais.

## Por que QLoRA?

Permite treinar uma LLM local reduzindo VRAM sem fazer full fine-tuning. Os pesos base permanecem quantizados/congelados, enquanto matrizes LoRA são treinadas.

## Experimentos controlados

### Experimento 01 — baseline

- LR: `2e-4`
- epochs: `2`
- LoRA rank: `8`
- LoRA alpha: `16`
- dropout: `0.05`
- effective batch: `8`
- max length: `1024`

### Experimento 02 — maior capacidade do adapter

Igual ao experimento 01, exceto:

- rank: `16`
- alpha: `32`

Pergunta: aumentar a capacidade do adapter melhora validação e benchmark?

### Experimento 03 — learning rate menor

Igual ao experimento 02, exceto:

- LR: `1e-4`

Pergunta: uma atualização mais conservadora melhora generalização?

## O que fica constante

- dataset;
- split;
- seed;
- epochs;
- effective batch;
- max length;
- dropout;
- modelo base;
- estratégia de quantização.

Assim podemos afirmar qual variável mudou entre pares de experimentos.

## Precisão

O script seleciona:

- BF16 quando `torch.cuda.is_bf16_supported()` retorna `True`;
- FP16 caso contrário.

Essa decisão é registrada em `metrics.json` e `experiments/results.csv`.

## GPU

`train.py` aborta caso `torch.cuda.is_available()` seja `False`.

## Evidências geradas

Cada run cria:

```text
experiments/experiment_XX/runs/<timestamp>/
  config_used.yaml
  environment.json
  metrics.json
  training.log
  checkpoints/
  adapter/
```

Além disso, o adapter mais recente é copiado para:

`experiments/experiment_XX/adapter/`

Os pesos são ignorados pelo Git por padrão.

`training.log` duplica a saída exibida no terminal, sem esconder o progresso que
precisa aparecer no screenshot do VSCode. O `environment.json` registra versões,
GPU, configuração efetiva e hashes SHA-256 de treino/validação.

## Não realizado até execução na máquina

O repositório não contém valores fabricados de:

- training loss;
- validation loss;
- duração;
- VRAM;
- benchmark final.

Esses valores só aparecem depois de uma execução real.
