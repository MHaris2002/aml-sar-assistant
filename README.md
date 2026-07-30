# AML SAR Assistant

An end-to-end AI system that detects potentially fraudulent financial transactions, grounds its reasoning in real regulatory documents via RAG, and auto-drafts Suspicious Activity Reports (SARs) — with a live API, a Power BI dashboard, and a [companion mobile app](https://github.com/MHaris2002/aml-sar-mobile).

This project was built as a full-stack applied ML/AI portfolio piece, with an emphasis on honest, evidence-backed engineering: every design decision here was validated against real numbers, and every dead end is documented rather than hidden.

<<<<<<< HEAD
## System Flow

![Architecture Flow](screenshots/flow.png)

=======
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
## What it does

1. **Detects** fraudulent transactions from raw financial data using a supervised ML model
2. **Retrieves** relevant regulatory context from a knowledge base of real FinCEN/FATF documents (RAG)
3. **Reasons** with an LLM to classify the fraud typology and explain why, citing sources
4. **Drafts** a structured SAR narrative, formatted for regulatory review
5. **Fills its own knowledge gaps** — when retrieval confidence is weak, a separate background job searches trusted regulatory domains and ingests better source material automatically
6. **Visualizes** results in Power BI and a live mobile app

<<<<<<< HEAD
=======
## Architecture

```
PaySim dataset (6.3M transactions)
        │
        ▼
Feature engineering (balance-mismatch detection)
        │
        ▼
Random Forest classifier  ──────────────►  97.85% precision, 99.63% recall
        │
        ▼
RAG knowledge base (FinCEN + FATF PDFs, ChromaDB, sentence-transformers embeddings)
        │
        ▼
LLM orchestration (Groq / Llama 3.3 70B)
   ├─ Summarize transaction
   ├─ Retrieve + classify typology (grounded, hallucination-checked)
   └─ Draft SAR narrative
        │
        ▼
SQLite database ──► FastAPI backend ──► Power BI dashboard
                            │
                            └──► React Native mobile app
```

>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
## The real story (not just the happy path)

This project deliberately documents its failures and fixes, because that's where the actual engineering judgment shows:

<<<<<<< HEAD
- **Started with unsupervised anomaly detection (Isolation Forest)** — got 1.9% recall. Diagnosed why (feature scale dominance, and having labeled data but not using it), switched to supervised learning, got 99.63% recall.
=======
- **Started with unsupervised anomaly detection (Isolation Forest)** — got 1.9% recall. Diagnosed why (feature scale dominance, and having labeled data but not using them), switched to supervised learning, got 99.63% recall.
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
- **RAG initially returned "weak match" on every transaction** — investigated why, found the detection model catches ledger-arithmetic anomalies (an account-takeover-style pattern) while the initial knowledge base only covered network-style money laundering typologies. Added FinCEN's specific Account Takeover advisory (FIN-2011-A016) to close the gap.
- **Even with the right documents, retrieval still failed** — the technical/ledger-style language used to describe transactions didn't semantically match how real regulatory documents describe fraud (behaviorally: "credential theft," "sudden wire transfer," not "balance error"). Fixed by translating technical patterns into red-flag language before querying — validated with a side-by-side retrieval test (0/6 vs 6/6 hit rate).
- **Chunk-boundary content loss** — added chunk overlap and neighbor-expansion retrieval (pulling in adjacent chunks around strong matches), which improved the RAG "strong match" rate from 80% to 96% (20/25 → 24/25) on a 25-transaction validation batch.
- **Small local LLMs (Ollama, 3B params) hallucinated specific details** (fake case numbers, invented countries) even with grounding instructions. Switched to a larger cloud model (Groq/Llama 3.3 70B) with a stricter prompt and low temperature — hallucination eliminated in subsequent testing.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile money transaction dataset (6.3M rows, 0.13% fraud rate) designed to simulate real-world mobile money fraud patterns. EDA confirmed 100% of fraud cases occur in TRANSFER/CASH_OUT transaction types only, informing the filtering and feature engineering strategy.

## Knowledge base sources

- FATF: Money Laundering Using New Payment Methods (2010)
- FATF: Report on New Payment Methods (2006)
- FATF: Money Laundering and Terrorist Financing Typologies (2004-2005)
- FATF: Mutual Evaluation Report, Belgium (2025)
- FinCEN Advisory FIN-2011-A016: Account Takeover Activity
- FinCEN Advisory FIN-2016-A003: E-mail Compromise Fraud Schemes
- Plus documents auto-discovered by the gap-filling search layer (logged and tagged separately from curated sources)

## Tech stack

- **Data/ML:** pandas, scikit-learn (Random Forest), joblib
- **RAG:** ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`), pypdf
- **LLM:** Groq API (Llama 3.3 70B)
- **Search/ingestion:** Tavily API (domain-restricted to fincen.gov, fatf-gafi.org, occ.gov, federalreserve.gov)
- **Backend:** FastAPI, SQLite
- **Visualization:** Power BI
- **Mobile:** React Native / Expo (see [companion repo](https://github.com/MHaris2002/aml-sar-mobile))

## Repo structure

```
├── data/
│   ├── raw/                    # PaySim CSV (gitignored - see setup)
│   ├── model_outputs/          # Trained model, flagged transactions
│   ├── knowledge_base/         # AML source PDFs + Chroma vector store
│   ├── sar_outputs/            # Generated SAR reports (JSON)
│   └── aml_sar.db              # Consolidated SQLite database
├── scripts/
│   ├── eda.py                  # Exploratory data analysis
│   ├── feature_engineering_and_model.py   # Isolation Forest (documented failure)
│   ├── supervised_model.py     # Random Forest (successful approach)
│   ├── build_knowledge_base.py # PDF chunking + embedding
│   ├── llm_orchestration.py    # 3-role LLM pipeline (summarize/classify/draft)
│   ├── gap_filling_search.py   # Autonomous knowledge base expansion
<<<<<<< HEAD
│   ├── add_clear_transactions.py  # Adds legitimate transactions for dashboard/app balance
=======
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
│   ├── build_database.py       # SQLite consolidation
│   └── export_for_powerbi.py
├── backend/
│   ├── main.py                 # FastAPI app
│   └── pipeline.py             # Core pipeline logic, importable
├── dashboard/
│   └── AML_SAR_Dashboard.pbix  # 3-page Power BI report
<<<<<<< HEAD
├── docs/
│   └── architecture-flow.png
=======
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Add a `.env` file:
```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
<<<<<<< HEAD
HF_HUB_OFFLINE=1
=======
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
```

Download [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) into `data/raw/`.

Run the pipeline in order:
```bash
python scripts/eda.py
python scripts/supervised_model.py
python scripts/build_knowledge_base.py
python scripts/llm_orchestration.py
<<<<<<< HEAD
python scripts/add_clear_transactions.py
python scripts/build_database.py
```

Start the API (use `--host 0.0.0.0` so a mobile app on the same network can reach it):
=======
python scripts/build_database.py
```

Start the API:
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
```bash
uvicorn backend.main:app --reload --host 0.0.0.0
```

<<<<<<< HEAD
Interactive API docs available at `http://localhost:8000/docs`.

=======
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
## Model performance

| Approach | Precision | Recall |
|---|---|---|
| Dataset's naive rule (`isFlaggedFraud`) | — | 0.2% |
| Isolation Forest (unsupervised) | 1.93% | 1.92% |
| **Random Forest (supervised)** | **97.85%** | **99.63%** |

RAG match confidence (25-transaction validation batch):

| Retrieval method | Strong match | Weak match |
|---|---|---|
| Top-6 only | 80% (20/25) | 16% (4/25) |
| + Neighbor expansion | **96% (24/25)** | **4% (1/25)** |

## Known limitations

- PaySim's synthetic fraud pattern (account-drain + balance mismatch) is closer to real-world account-takeover fraud than classic money-laundering typologies — the RAG knowledge base reflects this
- 99%+ model recall reflects a synthetic, deliberately-injected fraud pattern; real-world fraud is messier and adversarial
- LLM-generated narrative occasionally contains minor arithmetic inconsistencies in its prose explanation, despite the underlying classification being correctly grounded — raw figures should always be checked against the source data, not just the narrative
- No authentication on the API — this is a demo/portfolio system, not production-ready for real financial data

## Related

<<<<<<< HEAD
- [Mobile app (aml-sar-mobile)](https://github.com/MHaris2002/aml-sar-mobile) — React Native client for this backend
=======
- [Mobile app (aml-sar-mobile)](https://github.com/MHaris2002/aml-sar-mobile) — React Native client for this backend
>>>>>>> f370ea31f522d4c20b062952564e41ec87fe4e92
