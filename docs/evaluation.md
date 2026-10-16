# Avaliação

## Objetivo

Demonstrar com evidência mensurável se o fine-tuning especializado alterou a qualidade das respostas.

Comparação oficial:

`Qwen3-1.7B-Base` **vs** `Qwen3-1.7B-Base + adapter QLoRA`

Os dois modelos recebem:

- o mesmo `test.jsonl`;
- o mesmo prompt;
- `do_sample=False`;
- o mesmo `max_new_tokens`;
- o mesmo `repetition_penalty`.

## Test set

O test set possui 60 exemplos e não é lido por `train.py`.

## Métricas automáticas

`src/evaluate.py` calcula:

### Bottleneck term hit

Verifica se a resposta contém ao menos um termo esperado relacionado ao gargalo.

### Pattern term hit

Verifica se a resposta cita um padrão/abordagem esperada.

### Required concept coverage

Percentual de conceitos essenciais presentes na resposta.

### Format compliance

Percentual das oito seções esperadas preenchidas.

### Forbidden claim hit

Heurística que procura afirmações previamente marcadas como tecnicamente incorretas.

### ROUGE-L

Similaridade superficial entre a resposta e a referência.

### Heuristic auto score

Score transparente composto por:

```text
25% bottleneck hit
25% pattern hit
20% concept coverage
15% format compliance
15% ROUGE-L
-10% forbidden claim hit
```

Esse score serve somente para comparação interna. Não é tratado como benchmark universal.

## Limitações das métricas automáticas

Uma resposta pode ser correta com palavras diferentes e perder pontos em keyword matching.

ROUGE-L não mede verdade arquitetural.

Por isso o resultado final deve incluir avaliação humana.

## Rubrica humana 0–4

Cada resposta é avaliada em:

1. identificação correta do gargalo;
2. solução arquitetural;
3. justificativa técnica;
4. relevância;
5. ausência de alucinação;
6. completude;
7. clareza.

Escala:

- 0 = incorreto/ausente
- 1 = muito fraco
- 2 = parcialmente correto
- 3 = correto
- 4 = excelente

## Avaliação cega

O script produz:

`blind_review.csv`

As respostas são embaralhadas como `A` e `B`.

O arquivo:

`blind_key_DO_NOT_SHARE_WITH_REVIEWER.csv`

contém a identidade real de A/B e deve ficar escondido dos avaliadores.

Idealmente dois integrantes preenchem cópias independentes.

Depois:

```powershell
python src/score_manual_review.py `
  --review reviewer_1.csv reviewer_2.csv `
  --key experiments/evaluation/<RUN>/blind_key_DO_NOT_SHARE_WITH_REVIEWER.csv
```

## Casos em que piorou

`worsened_cases.csv` registra automaticamente casos nos quais o score automático do fine-tuned ficou abaixo do base.

Esses exemplos devem ser apresentados, não escondidos.

## Evidência final recomendada

Tabela:

| Métrica | Base | Fine-tuned | Delta |
|---|---:|---:|---:|
| Bottleneck hit | resultado real | resultado real | real |
| Pattern hit | real | real | real |
| Concept coverage | real | real | real |
| Format compliance | real | real | real |
| ROUGE-L | real | real | real |
| Human rubric | real | real | real |

Não preencher números antes da execução.
