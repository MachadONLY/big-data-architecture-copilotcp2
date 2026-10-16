"""Transparent, rule-based offline illustration of the frontend output format.

This module is deliberately NOT an LLM, fine-tuning, CUDA inference or an
experimental result. It makes the UI examinable before the GPU experiments.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLUEPRINTS = ROOT / "data" / "raw" / "scenario_blueprints.json"

KEYWORDS: dict[str, tuple[str, ...]] = {
    "n_to_m_integrations": ("ponto a ponto", "n x m", "n×m", "integracoes diretas", "integracao direta", "cada consumidor", "adicionar um consumidor", "novos consumidores"),
    "consumer_lag": ("consumer lag", "consumidor lento", "consumidores lentos", "atraso dos consumidores", "backlog", "lag"),
    "cdc": ("change data capture", "cdc", "alteracoes no banco", "capturar alteracoes", "outbox"),
    "partitioning": ("particionamento", "particoes", "particao", "hot partition", "chave de particao"),
    "schemas": ("schema registry", "schema evolution", "esquema de eventos", "schema", "avro", "protobuf", "compatibilidade de contratos"),
    "retention": ("retencao", "prazo de retencao", "eventos expiram", "historico de eventos"),
    "replay": ("replay", "reprocessar eventos", "reprocessamento de eventos", "reler eventos", "eventos antigos"),
    "oltp_vs_olap": ("oltp", "olap", "consultas analiticas", "banco transacional", "transacoes e relatorios"),
    "batch_vs_streaming": ("batch", "streaming", "tempo real", "pipeline noturno", "minutos de atraso", "processamento em lote"),
    "queues_vs_logs": ("rabbitmq", "message queue", "fila tradicional", "filas vs logs", "fila versus log"),
    "data_lake": ("data lake", "lago de dados", "armazenamento bruto"),
    "data_warehouse": ("data warehouse", "warehouse", "armazem de dados", "business intelligence"),
    "lakehouse": ("lakehouse", "delta lake", "iceberg", "hudi"),
    "etl_vs_elt": ("etl", "elt", "transformacao dos dados", "transformar antes de carregar"),
    "availability": ("disponibilidade", "alta disponibilidade", "failover", "downtime", "indisponibilidade"),
    "replication": ("replicacao", "replicas", "fator de replicacao", "replication"),
    "distributed_processing": ("spark", "hadoop", "mapreduce", "processamento distribuido", "flink"),
    "consistency_tradeoffs": ("consistencia", "consistency", "consistencia eventual", "exactly once", "idempotencia"),
    "scalability": ("escalabilidade", "escalar", "throughput", "taxa de eventos", "volume crescente"),
    "kafka_event_streaming": ("kafka", "redpanda", "event streaming", "event-driven", "producer", "consumer", "eventos"),
}


def normalize(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in norm if not unicodedata.combining(c))


def match_category(scenario: str) -> str | None:
    """Prefer high-signal phrases over isolated technology words."""
    text = normalize(scenario)
    scores: dict[str, int] = {}
    for category, keywords in KEYWORDS.items():
        scores[category] = sum((3 if " " in word else 1) * len(word)
                               for word in keywords if normalize(word) in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def illustrate(scenario: str) -> dict[str, str | None]:
    """Return example structured diagnostics, never pretend to run a model."""
    if not scenario.strip():
        raise ValueError("Informe um cenário de arquitetura.")
    category = match_category(scenario)
    if not category:
        text = (
            "Gargalo: Não determinado apenas com a descrição fornecida.\n"
            "Causa raiz: Dados insuficientes para identificar a origem do problema.\n"
            "Padrão recomendado: Não há recomendação segura sem métricas e requisitos.\n"
            "Solução: Informar volume, SLA, consumidores, fonte de dados e falhas observadas.\n"
            "Justificativa técnica: A arquitetura depende das restrições do contexto.\n"
            "Tecnologias possíveis: A definir após o diagnóstico.\n"
            "Vantagens: Evita recomendar tecnologia sem evidências.\n"
            "Limitações/Trade-offs: Requer investigação do ambiente existente."
        )
        return {"category": None, "text": text, "mode": "illustrative_rules"}

    spec = json.loads(BLUEPRINTS.read_text(encoding="utf-8"))["categories"][category]
    text = (
        f"Gargalo: {spec['bottleneck']}\n"
        f"Causa raiz: {spec['root']}\n"
        f"Padrão recomendado: {spec['pattern']}\n"
        f"Solução: {spec['solution']}\n"
        f"Justificativa técnica: Recomendação ilustrativa baseada em correspondência "
        f"de termos da categoria {spec['title']}; é necessário validar o contexto específico.\n"
        f"Tecnologias possíveis: {', '.join(spec['techs'])}.\n"
        f"Vantagens: {', '.join(spec['advantages'])}.\n"
        f"Limitações/Trade-offs: {', '.join(spec['tradeoffs'])}."
    )
    return {"category": category, "text": text, "mode": "illustrative_rules"}


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description="Example rules-only output (NOT a trained LLM).")
    cli.add_argument("--cenario", default="Temos 8 produtores e 12 consumidores conectados ponto a ponto. Adicionar um consumidor exige alterar varios sistemas.")
    args = cli.parse_args()
    print("MODO ILUSTRATIVO SEM LLM / GPU — NAO E RESULTADO EXPERIMENTAL\n")
    print(illustrate(args.cenario)["text"])
