# Fine-tuning do Llama 3.1-8B com LoRA (Unsloth) em dados de produtos da Amazon

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lrzcalmon/llama31-finetuning-lora/blob/main/Tech_Challenge_03_Llama_3_1_8B.ipynb)

Pipeline completo de ajuste fino supervisionado (SFT) de um LLM de 8 bilhões de parâmetros numa GPU gratuita (Tesla T4, 15 GB), usando quantização 4-bit e LoRA. Projeto do Tech Challenge 03 da pós-graduação **IA para Devs (FIAP)**.

**Objetivo:** adaptar o Llama 3.1-8B para responder perguntas sobre produtos a partir do título, usando o dataset [LF-Amazon-1.3M](http://manikvarma.org/downloads/XC/XMLRepository.html).

## Como funciona

```
LF-Amazon-1.3M (JSONL)
   │  limpeza: remove controle e espaços, vazios e duplicados
   ▼
250.000 pares instrução / entrada / saída (formato Alpaca)
   │
   ▼
Llama 3.1-8B em 4-bit (Unsloth) + adaptadores LoRA (r=16)
   │  SFTTrainer (TRL), AdamW 8-bit, lr 2e-4
   ▼
Adaptadores salvos no Google Drive  →  inferência com o modelo ajustado
```

## Destaques técnicos

| Item | Escolha | Por quê |
| --- | --- | --- |
| Modelo base | `unsloth/Meta-Llama-3.1-8B` em 4-bit | Cabe nos 15 GB de uma T4 gratuita |
| Técnica | LoRA, rank 16, nas camadas q/k/v/o e MLP | Treina só 41,9 M de 8 B parâmetros (0,52%) |
| Biblioteca | Unsloth + TRL `SFTTrainer` | Treino cerca de 2x mais rápido e com menos memória |
| Otimizador | AdamW 8-bit, scheduler linear, warmup 5 | Menor consumo de VRAM |
| Dados | 250 mil exemplos de treino, 2.500 de teste | Amostra reprodutível (`random_state=42`) |

## Resultados e limitações

Esta versão é uma **prova de conceito do pipeline**, executada com `max_steps = 60` para caber no tempo da entrega:

- O treino rodou em 98 s, com loss final de 0,04, mas cobriu apenas cerca de 120 exemplos (0,05% de uma época).
- Com esse volume, o modelo ajustado ainda gera respostas degeneradas (repetição de tokens). O próximo passo é treinar por pelo menos uma época e comparar respostas antes e depois no conjunto de teste.

Próximos passos:

- [ ] Treinar por 1 época completa (ou mais passos) e registrar a curva de loss
- [ ] Avaliar no conjunto de teste (ROUGE/BLEU e exemplos lado a lado)
- [ ] Usar o template de chat do Llama 3.1 Instruct
- [ ] Publicar os adaptadores no Hugging Face Hub

## Como executar

1. Abra o notebook no Colab pelo botão acima e selecione GPU T4.
2. Baixe o LF-Amazon-1.3M e coloque `trn.json` e `tst.json` no seu Google Drive.
3. Ajuste os caminhos na célula de configuração e execute as células em ordem.

## Stack

Python · PyTorch · Unsloth · Hugging Face Transformers, PEFT, TRL e Datasets · bitsandbytes · pandas · Google Colab

## Autor

**Ricardo Calmon**: líder técnico em IA aplicada e modernização de sistemas legados.
[LinkedIn](https://www.linkedin.com/in/ricardocalmon) · [ricardocalmon.com.br](https://ricardocalmon.com.br)
