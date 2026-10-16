# QUICKSTART — Windows + VSCode

Este arquivo é a sequência curta. Para detalhes, veja o README.

## 1. Primeiro: descobrir GPU

```powershell
nvidia-smi
```

## 2. Ambiente Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Instalar PyTorch CUDA

Use:

https://pytorch.org/get-started/locally/

Escolha a build compatível com a GPU/driver mostrados pelo `nvidia-smi`.

## 4. Dependências

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## 5. Provar GPU e validar repo

```powershell
python src/gpu_check.py --require-cuda --output assets/screenshots/gpu_environment.json
python src/verify_stack.py --require-cuda --output assets/screenshots/stack_environment.json
python -m pytest -q
python src/validate_dataset.py
```


## Opção automatizada após o ambiente estar instalado

Quando PyTorch CUDA e as dependências já estiverem funcionando, você pode executar:

```powershell
.\scripts\run_pipeline.ps1
```

Esse script valida GPU/dataset, roda os três experimentos, escolhe o adapter pelo **menor validation loss** (sem olhar o test set), executa o benchmark final, congela o ambiente e gera gráficos.

## 6. Rodar experimentos

```powershell
.\scripts\run_experiments.ps1
```

Em outro terminal, durante o treino:

```powershell
nvidia-smi -l 2
```

Tire o screenshot real no VSCode.

## 7. Avaliar Base vs Fine-tuned

Escolha o adapter com base nos resultados reais:

```powershell
python src/evaluate.py --adapter experiments/experiment_02/adapter
```

Substitua `experiment_02` se outro experimento for escolhido.

## 8. Frontend

```powershell
streamlit run app/app.py
```

Execute uma análise real e salve um screenshot legível como
`assets/screenshots/frontend_demo.png`.

## 9. Congelar ambiente e gerar gráficos

```powershell
python -m pip freeze > environment.lock.txt
python src/plot_results.py --evaluation-summary experiments/evaluation/<RUN>/summary.json
```

## 10. GitHub

Crie um repositório **vazio** na conta `MachadONLY`, nome
`big-data-architecture-copilot`, e então execute:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\publish_github.ps1
```

Repita o mesmo comando depois de adicionar os resultados e screenshots reais.
Veja `docs/github.md`.

## 11. Gate final antes de enviar

Depois dos experimentos, avaliação, push no GitHub e screenshots reais em
`assets/screenshots/gpu_training_vscode.png` e `assets/screenshots/frontend_demo.png`,
adicione o quarto integrante verdadeiro em `team_members.json` (três integrantes já estão cadastrados). Então rode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\submission_gate.ps1
```

O comando só termina com `SUBMISSION GATE: PASS` quando encontra os três
experimentos e logs, GPU CUDA/NF4, melhor adapter, benchmark, os dois screenshots,
4–5 integrantes e o commit atual publicado no GitHub. Ele também gera
`FINAL_REPORT.md` sem inventar métricas.

## 12. Gerar o ZIP final para o Teams

Depois que o gate estiver em `PASS`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package_submission.ps1
```

Após o gate PASS, o empacotamento gera também o **ZIP separado do adapter**. Envie ambos se o professor precisar fazer inferência sem repetir o treinamento.

O arquivo de código será:

`dist/big-data-architecture-copilot-submission.zip`

Ele inclui código, dataset, documentação, métricas, avaliação e evidências, mas exclui `.venv`, checkpoints, adapters e pesos gigantes.


## Observações de QA pré-entrega

- Use **Python 3.11, 64 bits**, preferencialmente. Testes estáticos não substituem treino CUDA.
- Execute scripts PowerShell pelo diretório raiz do projeto. Se a política bloquear `.ps1`, use: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1` (não mude a política global).
- Se a placa exigir o fallback, rode: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -ModelOverride Qwen/Qwen3-0.6B-Base`.
- O app só realizará inferência se existir um adapter treinado em `experiments/*/adapter/` ou indicado por `COPILOT_ADAPTER`. Esse adapter **não está no ZIP leve**.
- Sem GPU CUDA, os testes estáticos podem passar, mas o projeto não cumpre o treinamento local exigido.
