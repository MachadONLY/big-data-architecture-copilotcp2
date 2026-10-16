# QA REPORT — 15/10/2026 — REVISÃO PRÉ-ENTREGA

## O que foi realmente executado neste ambiente de auditoria

- Integridade CRC do ZIP auditado originalmente: PASS (`zipfile.testzip() = None`).
- Suite de testes Python depois das correções: **18 passed**.
- Lint com Ruff: PASS.
- Contratos reais de API validados para `SFTConfig`, `BitsAndBytesConfig` e o stack de versões fixadas.
- Renderização inicial do frontend validada com `streamlit.testing.v1.AppTest`: PASS.
- Compilação sintática de `src`, `app` e `tests`: PASS.
- Pré-validação offline (`python src/qa_static.py`): PASS.
- Dataset regenerado com o mesmo código: PASS.
- Dataset validado: 400 registros; treino 280 / validação 60 / teste 60; 20 categorias; zero intersecção de IDs e zero repetição exata de `input`, `prompt` ou `output` entre splits.
- Revisão de fluxo/contratos realizada em `train.py`, `evaluate.py`, `inference.py`, `app.py`, scripts PowerShell e empacotamento.
- Submission Gate avaliado em projeto sem treino: **NOT READY**, corretamente rejeitando as evidências ausentes.

## Correções aplicadas

1. Pipeline PowerShell não segue ignorando códigos de falha de scripts Python.
2. Runner dos experimentos remove o redirecionamento `2>&1 | Tee-Object` que pode causar erros sob PowerShell e checa saída de cada treino.
3. Experimentos com `--model-override` registram a configuração efetivamente usada.
4. App resolve o modelo base lendo `adapter_config.json`, inclusive fallback 0.6B.
5. Avaliação recusa modelos base incompatíveis com o adapter e usa o modelo registrado no adapter.
6. HTML gerado pelo modelo é escapado antes de renderizar no Streamlit.
7. Interface não declara execução CUDA confirmada antes de um adapter estar disponível.
8. Submission Gate exige 3 IDs de experimentos distintos, métricas numéricas, adapter real, 60 linhas de benchmark, JSON CUDA, screenshot PNG legível e 4–5 integrantes com RM.
9. Empacotamento produz ZIP de código e, depois de treino real, um ZIP separado do adapter para reprodução.
10. Pré-validação estática `scripts/preflight.ps1` e testes negativos de falhas adicionados.
11. Stack QLoRA fixado em versões exatas e verificador de compatibilidade adicionado; o PyTorch continua sendo instalado separadamente para corresponder ao driver/CUDA da máquina.
12. Cada treino grava log, hashes dos datasets, ambiente, diretório da execução e metadados reproduzíveis.
13. Gate final exige evidência NF4 em CUDA, screenshot do frontend e commit enviado ao GitHub com árvore limpa.

## O que NÃO foi testado e não pode ser atestado daqui

- Não existe `nvidia-smi` e CUDA GPU operacional neste runtime de QA.
- As dependências foram instaladas em um ambiente isolado de auditoria com PyTorch CPU. Isso permitiu verificar imports, contratos de API e renderização inicial do Streamlit, mas não substitui a instalação CUDA da máquina de treino.
- Nenhum fine-tuning real, cálculo de loss, benchmark com inferência do modelo ou screenshot real do aplicativo foi executado neste runtime.
- Os scripts `.ps1` passaram por revisão estática, não por execução em um PowerShell Windows (indisponível neste runtime).
- Nenhum repositório GitHub foi publicado por esta auditoria.
- Não foi feita avaliação humana de correção técnica dos 400 rótulos; são respostas sintéticas e altamente padronizadas (com variação de contexto), não um benchmark externo independente.

## Bloqueadores da entrega completa conforme rubrica

- **Treinamento real** em NVIDIA CUDA para 3 configurações distintas.
- **Avaliação real** do modelo Base × Fine-tuned na amostra de teste separada.
- **Print real** da GPU no VSCode durante treinamento.
- **Adapter salvo** e validado; frontend com inferência realizada ao menos uma vez.
- **Nomes/RMs** reais do grupo e **GitHub** publicado.

Somente com esses bloqueadores resolvidos a equipe pode alegar cumprimento integral dos requisitos. Um ZIP apenas com código não é evidência de treinamento concluído.

O ZIP final pré-GPU também foi extraído em uma pasta limpa e aprovado novamente em: integridade do arquivo, 18 testes, Ruff, compilação, QA estática, validação do dataset, contratos de API e `MANIFEST.sha256`.

## Pré-validação mais curta na máquina Windows

Abra o diretório do projeto no VSCode, use PowerShell e rode, depois de instalar dependências:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\preflight.ps1 -RequireGpu
```

Se o `nvidia-smi` falhar ou PyTorch não enxergar a GPU, pare antes de iniciar treinamento. A correção de driver/PyTorch precisa respeitar a GPU/Windows da máquina.

## Execução para fechar os itens experimentais

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1
```

Se a GPU não comportar 1.7B, o fallback documentado (todos os experimentos com o MESMO modelo base) é:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -ModelOverride Qwen/Qwen3-0.6B-Base
```

Estes comandos **não** são instantâneos. A duração depende de download, CUDA/VRAM e velocidade da GPU; não há execução prévia que permita estimá-la com confiança.

Ao final: avaliar manualmente, salvar screenshot real em `assets/screenshots/gpu_training_vscode.png`, testar `streamlit run app/app.py`, publicar GitHub e gerar pacote separado de adapter.


## Validação independente — pacote com integrantes (15/10/2026)

- Os integrantes Camila do Prado Padalino RM98316, Gabriel Teixeira Machado RM551570 e Felipe Cavalcante Bressane RM97688 foram adicionados a `team_members.json` e ao README.
- Quarto integrante ainda não informado; o gate continua impedindo status READY com somente três.
- Neste ambiente independente: **17 testes executados e aprovados** (`test_finalization.py`, `test_configs.py`, `test_dataset.py`, `test_prompts.py`), incluindo um novo teste de consistência do grupo.
- Os 2 testes restantes que importam Streamlit/TRL não foram reexecutados aqui porque este runtime não inclui essas dependências. O relatório anterior registra 18 testes no ambiente anterior, com dependências instaladas.
- Compilação sintática e QA offline: PASS.
- Validação do dataset: 400 cenários, 20 categorias, 280/60/60, sem repetição exata nem ID leakage: PASS.
- O gate lê integrantes automaticamente do arquivo JSON; os relatórios locais de status não entram no GitHub, evitando dirty working tree causado por sua própria regeneração.
- Manifesto do ZIP final será recalculado a partir dos bytes realmente empacotados, evitando hash desatualizado após o treino.
- CUDA, treinamento, benchmark e inferência com adapter continuam **não executados neste ambiente independente**.

- O helper `scripts/publish_github.ps1` foi incluído para simplificar commits e push na conta `MachadONLY`; requer repositório vazio criado na interface do GitHub e credenciais Git locais. **Não foi executado em PowerShell Windows neste ambiente.**
