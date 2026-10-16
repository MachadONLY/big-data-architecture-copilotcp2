# Protocolo de revisão do dataset

O dataset é sintético; portanto, antes de apresentar resultados como evidência acadêmica, a equipe deve validar uma amostra relevante.

## Prioridade 1 — Test set

Revisar **todos os 60 exemplos de `test.jsonl`**.

Motivo: são esses exemplos que determinam a conclusão Base vs Fine-tuned.

Para cada caso verificar:

- diagnóstico do gargalo;
- causa raiz;
- padrão recomendado;
- tecnologias citadas;
- trade-offs;
- ausência de afirmações absolutas incorretas.

## Prioridade 2 — Train/validation

Revisar no mínimo:

- 2 exemplos de treino por categoria = 40;
- 1 exemplo de validation por categoria = 20.

Total mínimo recomendado: 120 exemplos revisados incluindo o test set.

## Registro

Use:

`data/processed/manual_review_checklist.csv`

Campos:

- reviewer;
- technical_correctness_0_4;
- notes.

## Correções

Se uma referência estiver errada:

1. corrigir o blueprint em `data/raw/scenario_blueprints.json` ou a lógica de geração;
2. regenerar:
   `python src/prepare_dataset.py`;
3. executar:
   `python -m pytest -q`;
4. commitar a alteração;
5. reiniciar os experimentos caso o train/validation tenha mudado.

Não editar apenas o JSONL processado sem atualizar a fonte reproduzível.
