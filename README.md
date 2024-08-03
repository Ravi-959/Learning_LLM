# Understanding and building GPT-like model.
## My-first repository

Completed reading the book "Building a Large Language Model From Scratch" by Sebastian Raschka
as an LLM REsearcher intern at attentions.ai in Pune.

Recommended for beginners to understand basic concepts of LLM like Transformer architecture, decoder, enocoder,
pretraining, Top_k sampling, Temperature, Fine-tuning, Self-attention mechanism, Neural networks, Text-generation etc.

This repository contains the files that I used to understand and implement the code given in the book.
Most of the code will be similar to that of the book.

Some part of the code might be in comments which indicates there is a better version of it implemented next.

> tensor.py           -> Basics related to tensors, optimizers, auto grad calculation using pytorch framework.

> tokenizer.py        -> Tokenizing text manually without using BPE in tiktoken library

> att_mech.py         -> Classes related to self-attetion mechanism and Multi-head attention.

> base_gpt.py         -> Buliding a gpt-like model, inlcuing transformerblocks, Normalizing layers to train and load online weights.

> pre-training.ipynb  -> calculating loss, and training the model to update the parameters of model by minimizing loss.

> loading.ipnyb       -> How to load the pre-trained weights from Open AI's GPT in to your model.

> This gave me a little trouble, because I defined my own names and skipped some unnecessary layers, had to understand the structure of
loaded weights, and what they represent to complete it.
