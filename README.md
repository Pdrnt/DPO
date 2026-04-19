# Laboratório 08 - Alinhamento Humano com DPO

Projeto acadêmico para implementação de alinhamento humano com **DPO (Direct Preference Optimization)**, com foco em comportamento **HHH**:

- **Helpful**
- **Honest**
- **Harmless**

## Objetivo do projeto

Este repositório implementa um pipeline simples de alinhamento de modelo de linguagem com DPO, utilizando um dataset de preferências com pares de respostas:

- uma resposta **escolhida** (`chosen`), segura e alinhada;
- uma resposta **rejeitada** (`rejected`), inadequada ou insegura.

O objetivo é ajustar o modelo para favorecer respostas mais seguras, úteis e apropriadas, especialmente em cenários de segurança, fraude, manipulação, violência verbal e conduta inadequada.

## Requisitos atendidos

- Dataset em `.jsonl`
- Mais de 30 exemplos
- Chaves obrigatórias: `prompt`, `chosen`, `rejected`
- Foco em segurança, restrição e comportamento adequado
- Uso da biblioteca `trl`
- Uso de `DPOTrainer`
- Modelo ator e modelo de referência
- `beta = 0.1`
- Configuração de treino com economia de memória
- Treinamento executado
- Script de inferência/validação
- Entrega final versionável com tag `v1.0`

## Estrutura do projeto

```text
DPO/
├── data/
│   └── preferences.jsonl
├── outputs/
│   ├── .gitkeep
│   └── dpo_checkpoints/
├── scripts/
│   ├── inspect_dataset.py
│   ├── train_dpo.py
│   └── infer_validate.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   └── dpo_pipeline.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.sh
Ambiente

Este projeto foi preparado para execução com Python 3 e ambiente virtual venv.

Criar o ambiente
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Setup automatizado
chmod +x setup.sh
./setup.sh
Dependências principais
torch
transformers
datasets
trl
accelerate
bitsandbytes
sentencepiece
scipy
Dataset de preferências

O dataset utilizado está em:

data/preferences.jsonl

Cada linha do arquivo contém um objeto JSON com o seguinte formato:

{
  "prompt": "Instrução do usuário",
  "chosen": "Resposta segura e alinhada",
  "rejected": "Resposta inadequada ou insegura"
}

O conjunto foi construído com exemplos voltados a:

phishing,
fraude,
manipulação,
invasão,
sabotagem,
difamação,
tom corporativo inadequado,
condutas antiéticas.
Inspeção do dataset

Para validar a estrutura do dataset:

python scripts/inspect_dataset.py

O script verifica:

existência do arquivo;
leitura válida em .jsonl;
presença das chaves obrigatórias;
quantidade mínima de exemplos.
Modelo utilizado

Para manter o experimento leve e executável em máquina pessoal, foi utilizado o modelo:

sshleifer/tiny-gpt2

Esse modelo foi escolhido por ser pequeno e suficiente para demonstrar o pipeline completo do laboratório, embora não represente desempenho de alinhamento comparável a modelos maiores.

Pipeline DPO

O pipeline implementado usa:

modelo ator: modelo que recebe atualização de pesos;
modelo de referência: modelo congelado para comparação;
DPOTrainer da biblioteca trl;
dataset de preferências em formato prompt/chosen/rejected.
Explicação matemática do parâmetro beta

No DPO, o parâmetro beta controla a intensidade com que o modelo é pressionado a preferir a resposta chosen em relação à rejected, levando em conta também a diferença entre o modelo atual e o modelo de referência. Intuitivamente, o beta funciona como um regulador da força da otimização de preferência. Se ele for muito alto, o modelo pode se afastar agressivamente do comportamento original e perder fluência, estabilidade ou generalidade. Se for muito baixo, a preferência aprendida pode ficar fraca demais. Por isso, o beta = 0.1 atua como uma espécie de “imposto” moderado sobre mudanças excessivas: ele permite empurrar o modelo em direção às respostas preferidas, mas sem destruir totalmente a distribuição original aprendida pelo modelo base. Em termos práticos, o beta ajuda a equilibrar alinhamento e preservação da fluência.

Configuração de treino

A configuração foi feita com foco em economia de memória:

per_device_train_batch_size = 1
gradient_accumulation_steps = 4
num_train_epochs = 1

Quando há suporte adequado, o projeto tenta usar:

paged_adamw_32bit

Quando a máquina estiver em CPU sem suporte CUDA para bitsandbytes, o código faz fallback para:

adamw_torch

Isso foi necessário para manter compatibilidade prática no ambiente local.

Executando o treinamento
python scripts/train_dpo.py

Ao final do treino, os artefatos são salvos em:

outputs/dpo_checkpoints
Executando a validação por inferência
python scripts/infer_validate.py

O script compara a pontuação de duas respostas para um prompt de risco:

uma resposta segura;
uma resposta rejeitada.
Resultado experimental obtido

No experimento realizado com o modelo pequeno e apenas 1 época de treinamento, a validação foi executada corretamente, mas a resposta rejeitada ainda não foi totalmente suprimida pela resposta segura no teste escolhido.

Isso não invalida a implementação do laboratório, porque:

o dataset foi construído corretamente;
o pipeline DPO foi implementado;
o treino foi executado com sucesso;
a comparação por inferência foi realizada;
o resultado foi analisado de forma honesta.

Do ponto de vista acadêmico, isso mostra que a infraestrutura do alinhamento foi montada corretamente, mas que a eficácia do alinhamento depende também de fatores como:

tamanho e qualidade do modelo,
quantidade de dados,
número de épocas,
diversidade das preferências,
capacidade do modelo base.
Limitações do experimento

Este projeto tem algumas limitações intencionais para manter viabilidade em máquina pessoal:

uso de modelo pequeno;
poucas épocas de treino;
dataset reduzido;
validação simplificada por comparação de log-probabilidade.

Mesmo assim, o projeto atende ao objetivo de demonstrar o fluxo de alinhamento com DPO de ponta a ponta.

Como repetir a execução
1. Ativar ambiente
source venv/bin/activate
2. Validar dataset
python scripts/inspect_dataset.py
3. Treinar modelo
python scripts/train_dpo.py
4. Rodar inferência/validação
python scripts/infer_validate.py
Versionamento

O projeto foi organizado no estilo commit por commit, permitindo evolução progressiva e clara do pipeline para apresentação acadêmica e correção via GitHub.

Nota obrigatória sobre uso de IA

Partes geradas/complementadas com IA, revisadas por Pedro Lima.

Referência da atividade

Este projeto segue os requisitos do Laboratório 08 de Alinhamento Humano com DPO, incluindo dataset com prompt, chosen e rejected, uso de trl, DPOTrainer, beta = 0.1, treino, inferência/validação e versionamento final com tag v1.0.