# 🦜🔗 F_Langchain — LangChain Building-Block Experiments

> A small, honest sandbox of **LangChain primitives** — a FAISS-backed RAG chain and a memory-backed conversational chain — kept as clean reference snippets while I explore the framework.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-core%20%2B%20community-1c3c3c?logo=langchain)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20store-0467DF)](https://github.com/facebookresearch/faiss)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)](https://platform.openai.com/)

**Status:** 🌱 *Early experiments / learning repo.* Two compact, working builder functions plus an experiment log. This is intentionally a scratchpad of reusable LangChain patterns, **not** a packaged library — there is no CLI, test suite, or `requirements.txt` yet. The roadmap below is aspirational.

---

## 🎯 Aim

Learn LangChain by writing the **smallest correct version** of each core pattern, then keeping it as a copy-pasteable reference. The repo currently captures two of the most common building blocks: retrieval-augmented Q&A and stateful conversation.

## 📦 What's actually here

```
F_Langchain/
├── src/
│   ├── rag_chain.py       # build_rag_chain(docs, api_key) -> RetrievalQA
│   └── memory_chain.py    # build_chat_chain(api_key, use_summary) -> ConversationChain
└── docs/
    └── experiments.md     # running log of experiments + takeaways
```

### `src/rag_chain.py` — minimal RAG
Builds a [FAISS](https://github.com/facebookresearch/faiss) vector store from a list of raw strings using `OpenAIEmbeddings`, then wires a `RetrievalQA` chain over `ChatOpenAI` (`gpt-4o-mini`, `temperature=0`) with `k=4` retrieval and source documents returned.

```python
from src.rag_chain import build_rag_chain

qa = build_rag_chain(docs=["...your texts..."], api_key="sk-...")
result = qa.invoke({"query": "What does the spec say about voltage?"})
print(result["result"], result["source_documents"])
```

### `src/memory_chain.py` — conversation with memory
Builds a `ConversationChain` over `ChatOpenAI` (`gpt-4o-mini`, `temperature=0.7`) that switches between `ConversationSummaryMemory` (default) and `ConversationBufferMemory`.

```python
from src.memory_chain import build_chat_chain

chat = build_chat_chain(api_key="sk-...", use_summary=True)
chat.predict(input="Hi, remember my name is Fahil.")
chat.predict(input="What's my name?")
```

### `docs/experiments.md` — the log
A short notebook of findings, e.g. *FAISS + `text-embedding-3-small` works well for factual Q&A on technical docs*, *`ConversationSummaryMemory` wins for >10-turn chats*, and a planned **ReAct agent** with search / calculator / Python-REPL tools.

## 🧩 Tech Stack

`langchain` (+ `langchain-openai`, `langchain-community`), **FAISS** vector store, **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small`).

## 🚀 Getting Started

> No lockfile is committed yet — install the imports the snippets use:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu
export OPENAI_API_KEY=sk-...
python -c "from src.rag_chain import build_rag_chain; print('ok')"
```

## 🔭 Roadmap (planned, not yet built)

- [ ] **ReAct agent** with tools (search, calculator, Python REPL) — sketched in `experiments.md`, not yet implemented.
- [ ] Pin dependencies (`pyproject.toml` / `requirements.txt`) and add a smoke test.
- [ ] Swap in a local/offline LLM provider (Ollama) so the snippets run without an API key.
- [ ] Streaming responses and a tiny CLI to exercise each chain.
- [ ] Document-loader + chunking front-end for the RAG chain (currently takes raw strings).

## ✅ Conclusion

A deliberately small, transparent set of LangChain reference snippets. It does exactly what it claims — two working chains and a learning log — and the roadmap marks clearly what is aspiration versus what is built.
