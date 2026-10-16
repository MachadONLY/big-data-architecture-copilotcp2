# Verificação final independente — 15/10/2026

## Escopo

O objetivo foi reexaminar o código de treinamento e avaliação, as três configurações, os arquivos de dataset, a interface e a documentação, sem atribuir resultados experimentais inexistentes. O enunciado do professor é reproduzido em `docs/technical_spec.md` e detalhado em `docs/submission_checklist.md`.

## Testes efetivamente realizados neste ambiente

- `python -m pytest -q`: **21 passed, 2 skipped**. Os dois skips foram nas verificações de integração com Streamlit e o stack TRL/PEFT/bitsandbytes, não instalado neste ambiente.
- `python -m compileall -q src app tests`: aprovado.
- `python src/qa_static.py`: aprovado; sintaxe, campos essenciais, documentação e dataset.
- `python src/validate_dataset.py`: aprovado; 400 exemplos, 280/60/60, 20 categorias, nenhuma duplicação exata de IDs, input, prompt e output.
- `python src/offline_demo.py`: aprovado; devolve seções estruturadas a partir de correspondências de termos do blueprint. **Não há modelo neural envolvido nesse teste.**
- O ZIP final foi extraído e seus arquivos testados com `ZipFile.testzip` e manifesto SHA-256 no fechamento desta entrega.

## Análise de riscos que permanecem

| Requisito | Situação real | Consequência |
|---|---|---|
| Escolha de LLM / problema | Documentados | Inspecionável |
| Hiperparâmetros | 3 configs prontas; execução pendente | Sem comparação empírica |
| Avaliação | Código e rubrica; resultados pendentes | Sem Base × Fine-tuned medido |
| GPU / CUDA | Não acessível neste ambiente | Sem treino e prova de GPU |
| Interface | Código + demonstração ilustrativa offline | O modo ilustrativo **não é** uma resposta do modelo treinado |
| Print GPU no VSCode | Ausente | Critério oficial não atendido |
| GitHub | Código pronto; publicação não comprovada | Critério oficial não atendido |
| Equipe | 3 integrantes confirmados | Enunciado exige 4–5 |

Nenhum resultado de loss, CUDA, uso de VRAM, tempo de treinamento ou inferência de adapter foi fabricado. O funcionamento real dos scripts GPU permanece não verificado. O pacote não assegura pontuação mínima nem nota máxima.
