# Customer Satisfaction Classification using SCRDR

## Project Overview

ဒီ Project မှာ Single Classification Ripple Down Rules (SCRDR) စနစ်ကို အသုံးပြုပြီး Customer တွေရဲ့ စိတ်ကျေနပ်မှုကို Satisfied (ကျေနပ်မှုရှိ)၊ Neutral (ပုံမှန်) နဲ့ Dissatisfied (မကျေနပ်) ဆိုပြီး အတန်းအစား (၃) မျိုး ခွဲခြားနိုင်အောင် ဖန်တီးထားပါတယ်။  

SCRDR ဟာ ပုံမှန် Machine Learning တွေလို ကိန်းဂဏန်းအချက်အလက်တွေအပေါ်မှာပဲ အခြေခံတာမျိုး မဟုတ်ပါဘူး။ သူက ကျွမ်းကျင်သူတွေရဲ့ အသိပညာ (Knowledge-based) ကို အခြေခံတဲ့ နည်းပညာဖြစ်ပြီး လိုအပ်ချက်ရှိလာတဲ့အခါ Rule အသစ်တွေကို တဖြည်းဖြည်း ချဲ့ထွင်သွားတဲ့ စနစ်ဖြစ်ပါတယ်။ ခွဲခြားသတ်မှတ်မှုတိုင်းကို ဘယ် Rule ကြောင့် ဒီအဖြေထွက်လာတယ်ဆိုတာ အကြောင်းပြချက်နဲ့တကွ အတိအကျ ပြန်လည်စစ်ဆေးနိုင်ပါတယ်။  

---

## Problem Description

**ရည်ရွယ်ချက်:** Customer ဆီကရရှိတဲ့ တုံ့ပြန်ချက်တွေ (Feedback) ကိုကြည့်ပြီး သူတို့ရဲ့ စိတ်ကျေနပ်မှုအဆင့်ကို ခန့်မှန်းရန်။

**Input Features:**

| Feature             | Values                    | Why it matters                                         |
|---------------------|---------------------------|--------------------------------------------------------|
| `response_time`     | fast / medium / slow      | Faster response correlates with higher satisfaction    |
| `product_quality`   | high / medium / low       | Quality is a primary satisfaction driver               |
| `support_experience`| good / average / poor     | Support interaction shapes perception                  |
| `issue_resolved`    | yes / no                  | Unresolved issues strongly predict dissatisfaction     |
| `tone`              | positive / neutral / negative | Customer tone reflects overall experience           |
| `repeat_customer`   | yes / no                  | Loyal customers have higher tolerance                  |

**Output Labels:** `Satisfied`, `Neutral`, `Dissatisfied`

---

## What is SCRDR?

**Ripple Down Rules (RDR)** ဆိုတာ အသိပညာဗဟုသုတတွေကို စုဆောင်းတဲ့ နည်းပညာတစ်ခုဖြစ်ပြီး ရှိပြီးသား Rule တွေကို ပြင်စရာမလိုဘဲ ခြွင်းချက် (Exception) အသစ်တွေ ပေါင်းထည့်ရင်းနဲ့ စနစ်ကို ပိုကောင်းအောင် လုပ်ဆောင်တာဖြစ်ပါတယ်။

**Single Classification RDR (SCRDR)** extends RDR so that:

1. Rule Tree ကို ထိပ်ဆုံး (Root) ကနေ အောက်ခြေ (Leaf) အထိ လမ်းကြောင်းတစ်ခုတည်း အတိုင်း စစ်ဆေးသွားပါတယ်။
2. Node တစ်ခုချင်းစီမှာ Rule နဲ့ ကိုက်ညီရင် အဲ့ဒီ Rule ရဲ့ Exception List ထဲကို ဆက်ဆင်းသွားပါတယ်။
3. လမ်းကြောင်းတစ်လျှောက်မှာ နောက်ဆုံးကိုက်ညီခဲ့တဲ့ Rule ကပေးတဲ့ အဖြေကိုပဲ ရလဒ်အဖြစ် သတ်မှတ်ပါတယ်။
4. အကယ်၍ အဖြေမှားခဲ့ရင် နောက်ဆုံးကိုက်ညီခဲ့တဲ့ Rule အောက်မှာ ခြွင်းချက် Rule အသစ်တစ်ခု ပေါင်းထည့်လိုက်ရုံပါပဲ။ Rule အဟောင်းတွေကို လုံးဝပြင်စရာမလိုပါဘူး။

This guarantees:
- **Monotonic improvement**: အမှားတစ်ခုကို ပြင်လိုက်လို့ အရင်ကမှန်နေတဲ့ အချက်တွေ မှားမသွားနိုင်ပါဘူး။
- **Full auditability**: prediction တစ်ခုချင်းစီမှာ ဘယ် Rule ကြောင့် ဘယ်လို အကြောင်းပြချက်နဲ့ ဒီအဖြေထွက်လာတယ်ဆိုတာ သိနိုင်ပါတယ်။
- **Expert compatibility**: Rule တွေက လူတွေဖတ်လို့ရတဲ့ စာသားတွေဖြစ်လို့ နယ်ပယ်အလိုက် ကျွမ်းကျင်သူတွေနဲ့ ဆွေးနွေးပြင်ဆင်ရ လွယ်ကူပါတယ်။

---

## Project Structure

```

customer_satisfaction_scrdr/
├── data/
│   └── customer_satisfaction_dataset.csv   # 40-sample dataset
├── rules/
│   └── scrdr_rules.json                    # SCRDR rule tree (JSON)
├── src/
│   ├── scrdr_engine.py                     # Core SCRDR evaluation engine
│   ├── evaluator.py                        # Batch evaluation and metrics
│   ├── rule_updater.py                     # Incremental learning / corrections
│   └── utils.py                            # JSON I/O, tree printing, formatting
├── notebook/
│   └── scrdr_experiment.ipynb              # Full experiment notebook
└── results/
    └── results_analysis.txt                # Detailed results and observations
```

---

## Setup and Execution

### Prerequisites

- Python 3.8 or higher
- No external libraries required for the SCRDR engine (only the Python standard library)
- For the notebook: `pip install jupyter notebook` (optional)

### Running the SCRDR Engine
Demo စမ်းကြည့်ရန် - အချက်အလက်အားလုံးကို စစ်ဆေးပြီး ရလဒ်ထုတ်ပြမည်

```bash
cd customer_satisfaction_scrdr/src

# Quick demo: evaluate all records and print results
python3 -c "
import sys; sys.path.insert(0, '.')
from utils import load_rules_from_json, print_rule_tree
from evaluator import load_dataset, evaluate_dataset, print_metrics

root = load_rules_from_json('../rules/scrdr_rules.json')
records = load_dataset('../data/customer_satisfaction_dataset.csv')
result = evaluate_dataset(records, root, verbose=True)
print_metrics(result)
"
```

### Running the Notebook

```bash
cd customer_satisfaction_scrdr/notebook
jupyter notebook scrdr_experiment.ipynb
```

---

## Example Input / Output

**Input record:**
```python
{
  "response_time": "fast",
  "product_quality": "high",
  "support_experience": "good",
  "issue_resolved": "yes",
  "tone": "positive",
  "repeat_customer": "yes"
}
```

**Output:**
```
Prediction : Satisfied
Path       : R0 → R1
Steps:
  [R0] conditions=(always) → conclusion=Neutral
    Justification: Default rule: no specific signals detected — classify as Neutral
  [R1] conditions=(tone=positive, issue_resolved=yes) → conclusion=Satisfied
    Justification: Positive tone combined with resolved issue strongly indicates satisfaction
```

---

## Results Summary

| Metric             | Value  |
|--------------------|--------|
| Accuracy (initial) | 82.5%  |
| Satisfied F1       | 0.857  |
| Neutral F1         | 0.632  |
| Dissatisfied F1    | 0.909  |
| Rules (initial)    | 14     |

After one incremental learning pass: **~92.5% accuracy** with 18 rules.

---

## Key Design Decisions

- **Root rule (R0)** always fires and defaults to `Neutral` — the safest middle ground.
- **Exception order matters**: earlier exceptions represent more general overrides; later ones are more specific corrections.
- **No feature engineering**: all features are categorical, matching real-world survey data.
- **No external dependencies**: the engine runs with pure Python standard library.
