# Seleção do modelo

## Opção principal

`Qwen/Qwen3-1.7B-Base`

Características documentadas no model card oficial:

- causal language model;
- 1.7B parâmetros;
- ~1.4B parâmetros sem embeddings;
- 28 camadas;
- 16 attention heads de query / 8 KV heads;
- contexto de 32.768 tokens;
- Apache 2.0.

## Fallback

`Qwen/Qwen3-0.6B-Base`

Características:

- 0.6B parâmetros;
- ~0.44B sem embeddings;
- 28 camadas;
- 32.768 tokens de contexto;
- Apache 2.0.

## Por que 1.7B como primeira tentativa?

### Capacidade

1.7B oferece mais capacidade do que 0.6B para representar relações entre sintomas arquiteturais, causas e trade-offs.

Isso não é tratado como prova automática de qualidade: o benchmark final decide se o resultado é suficiente.

### Execução local

O tamanho ainda está em uma faixa apropriada para experimentar QLoRA local, ao contrário de modelos muito maiores que aumentariam significativamente VRAM e tempo de treino.

### Licença

Apache 2.0 facilita uso acadêmico e redistribuição do código/configuração sem depender de API fechada.

### Base vs Instruct

Foi escolhido `Base` porque o objetivo acadêmico é observar o efeito do SFT. Um modelo Instruct já chega com comportamento de seguir instruções fortemente treinado.

## Estimativa de memória dos pesos

Somente para ordem de grandeza, ignorando metadados, buffers, activations e overhead:

### 1.7B

FP16/BF16 bruto:

```text
1.7 bilhões × 2 bytes ≈ 3.4 GB
```

4-bit bruto:

```text
1.7 bilhões × 0.5 byte ≈ 0.85 GB
```

### 0.6B

FP16/BF16 bruto:

```text
0.6 bilhões × 2 bytes ≈ 1.2 GB
```

4-bit bruto:

```text
0.6 bilhões × 0.5 byte ≈ 0.30 GB
```

Esses números **não são previsão de VRAM total do treinamento**.

QLoRA também consome memória com:

- activations;
- LoRA parameters;
- gradients;
- optimizer state;
- CUDA kernels/workspaces;
- KV/intermediate tensors;
- framework overhead.

Por isso o projeto mede peak VRAM real com PyTorch em vez de prometer uma quantidade antes de conhecer a máquina.

## Decision gate

1. executar `nvidia-smi`;
2. instalar PyTorch CUDA compatível;
3. validar `torch.cuda.is_available()`;
4. fazer um run do 1.7B;
5. se ocorrer OOM mesmo com batch 1, gradient accumulation e max length 1024:
   - confirmar que não existem outros processos ocupando VRAM;
   - reduzir `max_length` de forma documentada, se o comprimento real do dataset permitir;
   - se continuar impraticável, migrar **todos** os experimentos para `Qwen3-0.6B-Base`.

Não misturar 1.7B e 0.6B dentro da mesma comparação de hiperparâmetros.

## Tempo de treino

Não há tempo inventado no documento.

O script mede `train_runtime_seconds` em cada experimento. Esse valor deve ser apresentado como resultado empírico da GPU usada.
