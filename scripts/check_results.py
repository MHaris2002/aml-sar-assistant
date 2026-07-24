import json
from collections import Counter
from pathlib import Path

results_path = Path("data/sar_outputs/sample_sar_results.json")
data = json.loads(results_path.read_text())

print(f"Total results: {len(data)}")

categories = []
match_strength = []

for r in data:
    analysis = r["typology_analysis"]
    if "ACCOUNT TAKEOVER" in analysis:
        categories.append("ACCOUNT TAKEOVER")
    elif "MONEY LAUNDERING" in analysis:
        categories.append("MONEY LAUNDERING")
    else:
        categories.append("UNCLEAR")

    if "weak" in analysis.lower():
        match_strength.append("weak")
    elif "strong" in analysis.lower():
        match_strength.append("strong")
    else:
        match_strength.append("moderate/unspecified")

print("\nCategory breakdown:")
for cat, count in Counter(categories).items():
    print(f"  {cat}: {count}")

print("\nMatch strength breakdown:")
for strength, count in Counter(match_strength).items():
    print(f"  {strength}: {count}")