# Benchmark

O benchmark oficial é o `data/processed/test.jsonl` (60 exemplos).

Ele não deve ser usado por `train.py`.

`src/evaluate.py` lê esse conjunto após o treinamento para comparar o modelo Base com o adapter escolhido.
