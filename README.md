# DomainSpecificAI_CS

This repository contains the code for finetuning and inferring a model for performing code search from natural language.

Requirements: Python 3.11

Instruction:

1) Dowloading datasets (CodeSearchNet) and fine-tuned models from this site: https://huggingface.co/datasets/hungphd/RP_CS/tree/main
2) Finetuning UniXcoder model (if you don't have computation resources, you can use the fine-tuned models from step 1): run finetune_search_train.sh.
3) Performing code search with logged vector for Natural Language and code: run finetune_search_gen.sh.
4) Designing prompt for reranking code output (TBD).
5) Running Reranking with LLM (TBD).
