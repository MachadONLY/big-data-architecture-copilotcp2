# Publicação no GitHub

## Antes do primeiro push

Confirme que:

```powershell
git status
```

não mostra:

- `.venv/`
- checkpoints;
- adapters;
- `.safetensors`;
- `.bin`;
- outros pesos grandes.

## Criar repositório local

```powershell
git init
git add .
git commit -m "feat: big data architecture copilot baseline"
```

Crie um repositório vazio no GitHub e depois:

```powershell
git branch -M main
git remote add origin <URL_DO_REPOSITORIO>
git push -u origin main
```

O gate final compara o `HEAD` local com `refs/heads/<branch>` no `origin` e exige
uma árvore de trabalho limpa. Portanto, depois de adicionar as evidências finais,
faça novo commit e `git push` antes de empacotar.

## Após cada experimento real

Os pesos continuam ignorados, mas métricas leves podem ser versionadas:

```powershell
git add experiments/results.csv
git add assets/charts/
git add assets/screenshots/gpu_environment.json
git commit -m "exp: record real QLoRA experiment results"
git push
```

Se quiser incluir um screenshot PNG ignorado por padrão:

```powershell
git add -f assets/screenshots/gpu_training_vscode.png
```

## Antes da apresentação

```powershell
python -m pytest -q
python src/validate_dataset.py
python src/gpu_check.py --require-cuda
git status
git log -1 --oneline
```

Registre o hash do commit apresentado.

## Não colocar no GitHub

- modelo Qwen completo;
- checkpoints grandes;
- adapters se o tamanho/política do repositório não justificar;
- `.venv`;
- caches;
- credenciais.

## Relatórios locais de finalização

`FINAL_REPORT.md` e `submission_status.json` são artefatos regenerados pelo gate e incluídos no ZIP do Teams. Eles não entram no Git por padrão, evitando alterações automáticas que invalidariam a verificação do último commit. Mantenha no GitHub os resultados medidos, logs, gráficos e evidências (sem pesos grandes).

## Publicação simplificada (PowerShell)

Depois de criar um repositório **vazio**, sem README inicial, em
`https://github.com/MachadONLY/big-data-architecture-copilot`, o mesmo comando
serve para o primeiro push e para os commits subsequentes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\publish_github.ps1
```

O comando usa a identidade informada pelo integrante, versão local do Git e a
autenticação GitHub já configurada na máquina. Se o Git solicitar login, conclua
a autenticação normal; não coloque tokens em arquivos do projeto.
