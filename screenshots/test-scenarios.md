# Test Scenarios & Results

Manual validation of the `/analyze` endpoint and mobile app, covering both
expected-clear and expected-fraud cases across a range of transaction patterns.
Run against the live pipeline (Random Forest model → RAG retrieval → LLM
classification → SAR draft) via the React Native app's "Check New" screen.

## Simple mode (natural transfer amounts)

| # | Scenario | Amount | Sender Before → After | Recipient Before → After | Expected | Actual |
|---|---|---|---|---|---|---|
| 1 | Normal transfer, balances intact | $500 | $5,000 → $4,500 | $2,000 → $2,500 | Clear | ✅ Clear |
| 2 | Full drain, new recipient | $5,000 | $5,000 → $0 | $0 → $5,000 | Flagged | ✅ Flagged — Account Takeover (95% confidence) |
| 3 | Large but proportionate | $50,000 | $500,000 → $450,000 | $100,000 → $150,000 | Clear | ✅ Clear |

## Advanced mode (exact ledger patterns from EDA findings)

| # | Scenario | Amount | Sender Before → After | Recipient Before → After | Expected | Actual |
|---|---|---|---|---|---|---|
| 4 | Classic account takeover (destination never updates) | $87,622.50 | $87,622.50 → $0 | $0 → $0 | Flagged, strong match | ✅ Flagged — Account Takeover, strong match, cites FIN-2011-A016 |
| 5 | Balance surplus anomaly (destination receives more than transferred) | $143,332 | $143,332 → $0 | $482,652.15 → $716,367.34 | Flagged, weak/ambiguous match | ✅ Flagged, weaker typology match — correctly identified as ambiguous |

## What these tests confirm

- **No false positives on ordinary transfers** (#1, #3) — the model doesn't just react to large dollar amounts; a $50,000 transfer that leaves the sender with a healthy balance is correctly left unflagged, consistent with the EDA finding that amount alone is a weak fraud signal.
- **The core fraud fingerprint is reliably caught** (#2, #4) — full account drains, especially paired with a destination account that doesn't reflect the incoming funds, are the strongest and most consistent trigger, matching the feature importances from model training (`amount_to_orig_balance_ratio` and `orig_balance_error` were the top two features).
- **The system is honest about ambiguity** (#5) — when a transaction is flagged but doesn't cleanly match either typology category, the LLM reports a weaker match rather than forcing false confidence. This is intentional behavior from the RAG grounding design, not a bug.

## How to reproduce

1. Start the backend: `uvicorn backend.main:app --reload --host 0.0.0.0`
2. Open the mobile app, go to **Check New**
3. For scenarios 1–3: use the default form fields (amount, your balance, recipient's balance)
4. For scenarios 4–5: toggle **Advanced mode** to set exact before/after balances
5. Tap **Check This Transfer** and compare the result against the table above