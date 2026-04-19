# Laboratório 08 - Alinhamento Humano com DPO

Projeto acadêmico para implementação de alinhamento humano com DPO (Direct Preference Optimization), com foco em comportamento HHH:

- Helpful
- Honest
- Harmless

## Objetivo

Construir um pipeline simples e organizado que:

1. utilize um dataset de preferências em `.jsonl`,
2. treine um modelo com `trl` e `DPOTrainer`,
3. use `beta = 0.1`,
4. demonstre em validação que respostas inadequadas foram suprimidas em favor de respostas seguras.

## Estrutura inicial

```text
lab-dpo-alinhamento/
├── data/
├── outputs/
├── scripts/
├── src/
├── .gitignore
├── README.md
├── requirements.txt
└── setup.sh
Ambiente

Este projeto usa Python 3 e ambiente virtual venv.

Criar e instalar dependências
chmod +x setup.sh
./setup.sh
Ativar o ambiente manualmente depois
source venv/bin/activate
Dependências principais
torch
transformers
datasets
trl
accelerate
bitsandbytes
Status

Projeto em construção no estilo commit por commit.

Observações

A documentação final será expandida nas próximas etapas, incluindo:

explicação matemática do parâmetro beta,
instruções de execução,
validação por inferência,
nota obrigatória sobre uso de IA.
Referência da atividade

Este projeto segue os requisitos do Laboratório 08 de Alinhamento Humano com DPO, incluindo dataset com prompt, chosen e rejected, uso de trl, DPOTrainer, treino, inferência e versionamento final com v1.0.


---

## Comandos no terminal

Dentro da pasta do projeto:

```bash
chmod +x setup.sh
./setup.sh