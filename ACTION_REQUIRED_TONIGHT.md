# Ação necessária hoje — concluir a entrega com GPU real

Este repositório está preparado e validado no nível de código, mas **a entrega ainda não está pronta**. Os resultados experimentais, adapters, métricas e screenshots precisam ser gerados no computador NVIDIA da equipe. Não substitua essas evidências por valores inventados.

## 1. Abrir no PC com GPU NVIDIA

Extraia o ZIP, abra a pasta raiz no VSCode e execute no PowerShell integrado:

```powershell
nvidia-smi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instale primeiro o PyTorch indicado pelo seletor oficial para a versão CUDA do computador. Depois:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\preflight.ps1 -RequireGpu
```

O preflight só passa se CUDA, a GPU e a quantização NF4 do `bitsandbytes` funcionarem de verdade.

## 2. Identificar o grupo

Três integrantes já foram cadastrados em `team_members.json`:

- Camila do Prado Padalino — RM98316
- Gabriel Teixeira Machado — RM551570
- Felipe Cavalcante Bressane — RM97688

**Falta um quarto integrante real** para atender a exigência de 4–5 integrantes.
Abra `team_members.json` e acrescente um objeto com `name` e `rm`.
O pipeline, o gate e o empacotamento lerão a lista automaticamente. Não invente integrantes.

## 3. Rodar os três experimentos e a avaliação

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1
```

Mantenha outro terminal do VSCode visível durante o treinamento:

```powershell
nvidia-smi -l 2
```

Tire um print durante uma etapa de treino, mostrando VSCode, processo Python, utilização e memória da GPU. Salve como:

```text
assets/screenshots/gpu_training_vscode.png
```

## 4. Demonstrar o frontend

Depois que `experiments/best_adapter.txt` existir:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_app.ps1
```

Abra o endereço mostrado pelo Streamlit, gere uma resposta com o adapter real e salve o print em:

```text
assets/screenshots/frontend_demo.png
```

## 5. Publicar no GitHub

Crie **um repositório vazio**, sem README inicial, na conta `MachadONLY` com o nome
`big-data-architecture-copilot`. Depois execute o comando abaixo na raiz do projeto:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\publish_github.ps1
```

Este comando cria o commit e faz push. Execute-o novamente depois de adicionar os
resultados, screenshots reais e o quarto integrante. Se o Git solicitar login,
faça a autenticação normalmente. Não publique pesos de modelo ou credenciais.

## 6. Gate e pacote final

Depois dos prints e do push:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\submission_gate.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package_submission.ps1
```

Somente entregue quando o terminal mostrar `SUBMISSION GATE: PASS`. Revise `FINAL_REPORT.md`, anexe os ZIPs de `dist/` e informe o link do GitHub.

## Evidências que devem existir ao final

- três linhas reais em `experiments/results.csv`;
- logs e adapters dos três experimentos;
- `experiments/best_adapter.txt`;
- benchmark Base vs Fine-tuned e casos de piora;
- `environment.lock.txt`;
- `assets/screenshots/gpu_environment.json`;
- `assets/screenshots/stack_environment.json`;
- `assets/screenshots/gpu_training_vscode.png`;
- `assets/screenshots/frontend_demo.png`;
- quatro ou cinco integrantes no relatório;
- commit enviado ao `origin`, sem alterações locais pendentes.
