# Checklist de entrega — CP2/CP3

| Requisito do professor | Evidência no projeto | Status antes de rodar |
|---|---|---|
| 1. Selecionar LLM | README + docs/architecture.md + model card | Pronto |
| 2. Problema específico | Big Data Architecture Copilot | Pronto |
| 3. Testar parâmetros | 3 YAMLs + results.csv | Código pronto; resultados pendentes |
| 4. Avaliar modelo | evaluate.py + blind review | Código pronto; métricas pendentes |
| 5. Testes em GPU | train.py exige CUDA | Execução pendente |
| Stack compatível | verify_stack.py + NF4 CUDA | Execução pendente |
| 6. Frontend | app/app.py | Pronto; precisa adapter treinado |
| 7. Print GPU VSCode | docs/gpu.md descreve evidência | Screenshot pendente |
| 8. GitHub completo | estrutura, docs, dataset e scripts | Pronto para criar repo |
| Fine-tuning real | QLoRA + SFTTrainer + adapters | Código pronto; execução pendente |
| Train/val/test | 280/60/60 sem overlap de IDs | Pronto |
| Comparar Base vs Fine-tuned | evaluate.py | Código pronto; execução pendente |
| Mostrar failures | worsened_cases.csv | Gerado após avaliação |

## Antes da entrega

- [ ] revisão humana do dataset registrada;
- [ ] `gpu_check.py` executado com CUDA;
- [ ] `verify_stack.py --require-cuda` passou, incluindo NF4;
- [ ] experimento 01 executado;
- [ ] experimento 02 executado;
- [ ] experimento 03 executado;
- [ ] `results.csv` contém somente valores reais;
- [ ] melhor adapter escolhido com base nos resultados;
- [ ] benchmark Base vs Fine-tuned executado;
- [ ] duas avaliações cegas, se possível;
- [ ] screenshots reais adicionados;
- [ ] `assets/screenshots/frontend_demo.png` mostra resposta real do adapter;
- [ ] README atualizado com GPU real e resultados;
- [ ] `pip freeze` salvo como `environment.lock.txt`;
- [ ] frontend demonstrado localmente;
- [ ] GitHub revisado sem pesos gigantes.
- [ ] commit atual enviado ao `origin` e working tree limpa;

## Gate automatizado

Antes de enviar, execute `scripts/submission_gate.ps1`. Não submeta enquanto o gate não retornar `PASS`.
