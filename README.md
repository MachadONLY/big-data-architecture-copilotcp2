# Big Data Architecture Copilot

> CheckPoint 2/3 — Inteligência Artificial — FIAP 4ESPV-2026  
> Professor: Francisco Elanio Bezerra  
> Entrega: 15/10/2026

LLM local especializada em diagnosticar problemas de arquitetura de dados e recomendar padrões, tecnologias e trade-offs. O projeto está preparado para demonstrar **Supervised Fine-Tuning real com QLoRA**, execução em GPU CUDA, comparação Base vs Fine-tuned e uma interface Streamlit sem API comercial. O status abaixo separa claramente o que já foi validado do que ainda precisa ser executado na GPU da equipe.

---

## Abrir a demonstração em Windows (duplo clique)

Para inspecionar a interface sem GPU, use **`INICIAR_DEMO_WINDOWS.bat`** na raiz do projeto
(apenas Python 3 e internet para instalar Streamlit na primeira execução).
O cenário já vem preenchido: clique em **Analisar arquitetura**.
Esse modo é **ilustrativo, baseado em regras**, e **não substitui o treinamento de LLM**.
Para avaliação rápida, leia também **`LEIA_PRIMEIRO.txt`**.

Repositório publicado: https://github.com/MachadONLY/big-data-architecture-copilotcp2

## Verificação rápida sem GPU (interface e código)

Para demonstrar a estrutura da aplicação **antes de executar o treinamento**, há um modo ilustrativo explícito, baseado em regras e nos blueprints do dataset. **Ele não é uma LLM nem substitui o fine-tuning CUDA exigido na atividade.**

No terminal, apenas com Python:

```powershell
python src/qa_static.py
python src/validate_dataset.py
python src/offline_demo.py
```

Para abrir a interface Streamlit sem instalar as bibliotecas pesadas da LLM:

```powershell
python -m pip install -r requirements-ui.txt
python -m streamlit run app/app.py
```

Sem adapter, a interface anuncia claramente **demonstração baseada em regras**. Após a execução real de `scripts/run_pipeline.ps1` em CUDA, o mesmo frontend carrega o adapter salvo e executa inferência de LLM local. A documentação do método e dos resultados pendentes permanece separada em `RELATORIO_ACADEMICO.md` e `FINAL_REPORT.md`.

## 1. Problema

Modelos gerais conhecem conceitos como Kafka, CDC, OLTP/OLAP, data lakes e processamento distribuído, mas não foram especializados para diagnosticar cenários de arquitetura seguindo um formato técnico consistente.

Exemplo de entrada:

> Temos vários serviços produzindo eventos e vários consumidores. Cada integração é feita ponto a ponto e adicionar um novo consumidor exige alterar vários sistemas.

Saída esperada:

- gargalo;
- causa raiz;
- padrão recomendado;
- solução;
- justificativa técnica;
- tecnologias possíveis;
- vantagens;
- limitações/trade-offs.

---


#Integrantes

Integrantes confirmados para esta entrega:

- Camila do Prado Padalino — RM98316
- Gabriel Teixeira Machado — RM551570
- Felipe Cavalcante Bressane — RM97688


## Quickstart e GitHub

- [`QUICKSTART.md`](QUICKSTART.md)
- [`docs/github.md`](docs/github.md)

---

## Referências

Veja [`docs/references.md`](docs/references.md).

Fontes principais:

- Qwen3 official model card
- Hugging Face TRL
- Hugging Face PEFT
- bitsandbytes
- PyTorch
- LoRA paper
- QLoRA paper

