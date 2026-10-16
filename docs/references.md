# Referências técnicas

Use estas fontes na documentação/apresentação.

## Qwen3-1.7B-Base

Official model card:

https://huggingface.co/Qwen/Qwen3-1.7B-Base

Pontos relevantes:

- 1.7B parâmetros;
- 1.4B non-embedding;
- 28 layers;
- context length 32,768;
- Apache 2.0.

## TRL — SFTTrainer

https://huggingface.co/docs/trl/sft_trainer

Pontos relevantes:

- suporte a prompt-completion datasets;
- `completion_only_loss`;
- integração com PEFT.

## PEFT — LoRA / QLoRA

https://huggingface.co/docs/peft/main/package_reference/lora

https://huggingface.co/docs/peft/developer_guides/quantization

Ponto relevante:

- QLoRA-style training pode aplicar LoRA em todas as linear layers com `target_modules="all-linear"`.

## bitsandbytes

https://huggingface.co/docs/bitsandbytes/en/installation

Pontos relevantes:

- quantização NF4/FP4;
- requisitos de hardware;
- builds para Windows x86-64 e NVIDIA CUDA.

## PyTorch

https://pytorch.org/get-started/locally/

Usar para selecionar a wheel CUDA correta conforme ambiente real.

## QLoRA paper

Dettmers et al. — QLoRA: Efficient Finetuning of Quantized LLMs

https://arxiv.org/abs/2305.14314

## LoRA paper

Hu et al. — LoRA: Low-Rank Adaptation of Large Language Models

https://arxiv.org/abs/2106.09685
