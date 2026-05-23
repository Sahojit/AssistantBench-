---
title: Llama OSS Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Llama OSS Assistant

A standalone OSS assistant powered by **Llama 3.1 8B Instant** via Groq's free inference API.

## Setup

Set the following secrets in your HF Space settings:

| Secret | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (free at console.groq.com) |

## Features

- Three-layer safety guardrails (keyword filter + system prompt + output PII redaction)
- Sliding 10-turn conversation memory
- Tool augmentation: calculator and datetime lookups
- LangFuse observability (optional — set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`)
