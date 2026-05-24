---
title: Llama OSS Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# Llama OSS Assistant

A standalone OSS assistant powered by **Llama 3.1 8B Instant** via Groq's free inference API.

## Features
- Multi-turn conversations with 10-turn memory window
- 3-layer safety guardrails (keyword filter + system prompt + PII redaction)
- Tool augmentation: calculator and datetime lookups
- LangFuse observability (optional)

## Setup

Set the following secrets in your HF Space settings:

| Secret | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — free at console.groq.com |
| `LANGFUSE_PUBLIC_KEY` | LangFuse tracing (optional) |
| `LANGFUSE_SECRET_KEY` | LangFuse tracing (optional) |
| `LANGFUSE_HOST` | `https://us.cloud.langfuse.com` (optional) |
