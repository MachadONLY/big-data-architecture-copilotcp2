# Dataset

## Objetivo

Ensinar o modelo a diagnosticar cenários de arquitetura de dados em um formato consistente:

- Gargalo
- Causa raiz
- Padrão recomendado
- Solução
- Justificativa técnica
- Tecnologias possíveis
- Vantagens
- Limitações/Trade-offs

## Composição

O dataset incluído no repositório possui **400 exemplos**, distribuídos igualmente entre 20 categorias:

1. Kafka / event streaming
2. batch vs streaming
3. OLTP vs OLAP
4. filas vs logs distribuídos
5. integrações N×M
6. CDC
7. consumer lag
8. particionamento
9. escalabilidade
10. retenção
11. replay
12. schemas
13. data lake
14. data warehouse
15. lakehouse
16. ETL vs ELT
17. disponibilidade
18. replicação
19. processamento distribuído
20. consistência e trade-offs

Cada categoria possui 20 cenários contextualizados em setores, escalas e restrições diferentes.

## Split

- Train: 280 (14 por categoria)
- Validation: 60 (3 por categoria)
- Test: 60 (3 por categoria)

Seed do split: `20261001`.

## Proveniência

O conjunto é **sintético e autoral para o projeto**, construído a partir de uma matriz versionada em:

`data/raw/scenario_blueprints.json`

Ele não foi coletado de conversas privadas nem extraído de um dataset proprietário.

O script:

`src/prepare_dataset.py`

recria deterministicamente os três splits e executa verificações de:

- campos obrigatórios;
- IDs duplicados;
- equilíbrio entre categorias;
- interseção de IDs entre train/validation/test.

O arquivo `manual_review_checklist.csv` existe para a equipe registrar uma revisão técnica humana antes da entrega.

Além de IDs disjuntos, a geração rejeita repetição textual exata de `input`,
`prompt` ou `output` entre quaisquer splits e grava SHA-256 de cada JSONL em
`dataset_qa.json`.

## Formato de um registro

```json
{
  "id": "partitioning-014",
  "category": "partitioning",
  "difficulty": "medium",
  "instruction": "...",
  "input": "...",
  "output": "...",
  "expected_bottleneck_terms": ["hot partition", "partição", "skew"],
  "expected_pattern_terms": ["particionamento", "chave", "hash"],
  "must_include": ["partição", "chave", "cardinalidade", "ordenação"],
  "must_not_include": ["mais partições garantem ordenação global"],
  "prompt": "...",
  "completion": "..."
}
```

Os campos de avaliação não são uma resposta secreta passada ao modelo; são usados depois para o benchmark automático.

## Limitação acadêmica importante

Dataset sintético pode refletir simplificações ou vieses de quem o construiu. Por isso a submissão deve incluir revisão humana de uma amostra relevante e registrar correções no Git.

Os cenários são diferentes, mas permanecem **in-distribution**: exemplos da mesma
categoria compartilham conceitos arquiteturais e um blueprint de resposta. Isso
não é vazamento textual, porém pode favorecer consistência de formato e de
vocabulário. O trabalho não deve apresentar este test set como benchmark externo
independente.
