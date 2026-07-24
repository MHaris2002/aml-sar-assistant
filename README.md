# AML Transaction Monitoring & SAR Drafting Assistant

Fraud/AML detection pipeline on transaction data: lightweight anomaly detection,
RAG-grounded typology matching against real AML guidance, and LLM-drafted
Suspicious Activity Reports. Visualized in Power BI.

## Status
🚧 In progress

## Dataset
- [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

## Setup
1. `python -m venv venv` and activate it
2. `pip install -r requirements.txt`
3. Add API keys to `.env` (Kaggle, LLM provider)
4. `python scripts/download_data.py`
