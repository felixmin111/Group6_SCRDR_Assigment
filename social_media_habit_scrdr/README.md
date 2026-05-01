# Social Media Habit Analyzer using SCRDR

> **Single Classification Ripple Down Rules** နည်းပညာကို အသုံးပြု၍ Social media သုံးစွဲမှု ပုံစံများကို အမျိုးအစား ခွဲခြားပေးသည့် စနစ် ဖြစ်ပါသည်။

---

## Problem Description

Social media ကို အလွန်အကျွံ သုံးစွဲခြင်းသည် ယနေ့ခေတ်တွင် စိုးရိမ်စရာ ကိစ္စတစ်ခု ဖြစ်လာပါသည်။ ဤစနစ်သည် User တစ်ဦး၏ Social media အလေ့အထများကို အောက်ပါ အမျိုးအစား (၃) မျိုးထဲမှ တစ်ခုခုအဖြစ် ခွဲခြားပေးပါသည် -

- **Healthy** — လိုအပ်မှ သုံးခြင်း၊ ရည်ရွယ်ချက်ရှိရှိ သုံးခြင်းနှင့် အိပ်ချိန်မှန်ခြင်း
- **Moderate** — ပုံမှန်အတိုင်း သုံးစွဲခြင်း၊ အလေ့အထ အမျိုးမျိုး ရောနှောနေခြင်း
- **Unhealthy** — အလွန်အကျွံ သုံးစွဲခြင်း၊ အိပ်ချိန်မမှန်ခြင်းနှင့် စွဲလမ်းစွာ သုံးစွဲခြင်း

ဤစနစ်သည် ဘာကြောင့် ဤသို့ အဖြေထုတ်ပေးသည်ကို မသိနိုင်သော "Black-box ML Model" များကို အသုံးမပြုဘဲ **SCRDR (Single Classification Ripple Down Rules)** နည်းပညာကို အသုံးပြုထားပါသည်။ ၎င်းသည် ပွင့်လင်းမြင်သာပြီး အကြောင်းပြချက် ခိုင်လုံစွာ ရှင်းပြနိုင်သည့်အပြင် အချက်အလက်သစ်များကိုလည်း အမြဲ သင်ယူနိုင်သော စည်းမျဉ်းအခြေခံ (Rule-based) စနစ်တစ်ခု ဖြစ်ပါသည်။

---

## What is SCRDR?

SCRDR (Single Classification Ripple Down Rules) ဆိုသည်မှာ ဗဟုသုတများ ရှာဖွေစုဆောင်းခြင်းနှင့် အမျိုးအစား ခွဲခြားခြင်း နည်းပညာတစ်ခု ဖြစ်ပြီး အောက်ပါ ဂုဏ်သတ္တိများ ရှိပါသည် - 

1. **Single-path evaluation**: Input တစ်ခုအတွက် စည်းမျဉ်း လမ်းကြောင်း (Rule chain) တစ်ခုတည်းကိုသာ စစ်ဆေးပါသည်
2. **Exception-based refinement**: မှားယွင်းမှုများကို ပြင်ဆင်ရာတွင် ရှိပြီးသား စည်းမျဉ်းများကို မပြင်ဘဲ "ခြွင်းချက်" (Exception rules) များအဖြစ်သာ ထပ်တိုးသွားပါသည်
3. **Full explainability**: ဆုံးဖြတ်ချက် တိုင်းအတွက် မည်သည့် စည်းမျဉ်းကြောင့် ဖြစ်သည်ဟူသော ခိုင်လုံသော အကြောင်းပြချက်ကို ထုတ်ပေးနိုင်ပါသည်
4. **Incremental learning**: ဖြစ်ရပ်သစ်များနှင့် တွေ့ကြုံရသည်နှင့်အမျှ စည်းမျဉ်းသစ်များ ထပ်တိုးလာပြီး စနစ်က ပိုမို တော်လာပါသည်

**Traversal logic:**
```
For each rule in priority order:
  If rule matches input:
    Check exception child
    If exception matches: recurse deeper
    Else: return current conclusion
  Else: try next rule
```

---

## Project Structure

```
social_media_habit_scrdr/
├── data/
│   └── social_media_dataset.csv       # 45-sample dataset
├── rules/
│   └── scrdr_rules.json               # Rule tree (JSON)
├── src/
│   ├── scrdr_engine.py                # Core SCRDR evaluator
│   ├── evaluator.py                   # Batch evaluation + accuracy
│   ├── rule_updater.py                # Incremental learning
│   └── utils.py                       # Helpers, formatting
├── notebook/
│   └── scrdr_experiment.ipynb         # Full experiment walkthrough
├── results/
│   └── results_analysis.txt           # Analysis and observations
└── README.md
```

---

## Input Features

| Feature | Type | Values | Meaning |
|---------|------|--------|---------|
| `daily_usage_hours` | numeric | 0–10 | တစ်နေ့တာ ဆိုရှယ်မီဒီယာ သုံးစွဲချိန် (နာရီ) |
| `sleep_time` | categorical | early / normal / late | အိပ်ရာဝင်ချိန် အလေ့အထ |
| `primary_usage_type` | categorical | study / social / entertainment | အဓိက သုံးစွဲရခြင်း အကြောင်းရင်း |
| `notification_check_frequency` | categorical | low / medium / high | ဖုန်းခဏခဏ စစ်ဆေးမှု အခြေအနေ |
| `purpose` | categorical | learning / communication / scrolling | သုံးစွဲရသည့် ရည်ရွယ်ချက် |

**Output Labels:** `Healthy`, `Moderate`, `Unhealthy`

---

## How to Run

### Prerequisites

```bash
pip install pandas jupyter
```

### Option 1: Run the evaluator directly

```bash
cd src/
python evaluator.py
```

### Option 2: Interactive notebook

```bash
cd notebook/
jupyter notebook scrdr_experiment.ipynb
```

### Option 3: Python API

```python
import sys
sys.path.insert(0, 'src/')
from scrdr_engine import load_rules_from_json, evaluate_scrdr

rules = load_rules_from_json('rules/scrdr_rules.json')

user = {
    "daily_usage_hours": 6.5,
    "sleep_time": "late",
    "primary_usage_type": "entertainment",
    "notification_check_frequency": "high",
    "purpose": "scrolling"
}

result = evaluate_scrdr(user, rules)
print(result['label'])       # → Unhealthy
print(result['rule_path'])   # → ['R2']
print(result['explanation']) # → "High usage (>5h)..."
```

---

## Sample Output

```
[✓] ID: 1
    True Label : Healthy
    Predicted  : Healthy
    Rule Path  : R0 → R1
    Explanation: Low usage hours combined with good sleep schedule and purposeful usage
                 (learning or communication) indicates a Healthy habit.

[✓] ID: 22
    True Label : Unhealthy
    Predicted  : Unhealthy
    Rule Path  : R2
    Explanation: High usage (>5h), late-night sleep, and compulsive notification
                 checking indicate disruptive social media behavior.

============================================================
SCRDR EVALUATION SUMMARY
============================================================
Total Samples : 45
Correct       : 42
Accuracy      : 93.3%
Errors        : 3
============================================================
```

---

## Incremental Learning Example

အကယ်၍ စနစ်က အဖြေမှားထုတ်ပေးပါက၊ ခြွင်းချက် စည်းမျဉ်းအသစ်ကို အောက်ပါအတိုင်း ထပ်တိုးနိုင်ပါသည် -

```python
from rule_updater import correct_misclassification

new_rule = correct_misclassification(
    input_case=misclassified_input,
    true_label="Unhealthy",
    rules=rules,
    rules_filepath="rules/scrdr_rules.json",
    expert_conditions={
        "daily_usage_hours": {"op": ">=", "value": 4.5},
        "sleep_time": {"op": "==", "value": "late"},
        "notification_check_frequency": {"op": "==", "value": "high"}
    }
)
```

**Result:** ရှိပြီးသား စည်းမျဉ်းများကို မပြင်ဘဲ အဖြေမှားစေသော စည်းမျဉ်း၏ အောက်တွင် ခြွင်းချက်သစ် တစ်ခုအဖြစ် ထပ်တိုးသွားမည် ဖြစ်ပါသည်။

---

## Key Results

| Class | Precision | Recall |
|-------|-----------|--------|
| Healthy | 100% | 100% |
| Moderate | 83% | 100% |
| Unhealthy | 100% | 80% |
| **Overall** | | **93.3%** |

---

## Dependencies

```
python >= 3.9
pandas
jupyter (for notebook)
```

No external ML libraries required. The SCRDR engine is pure Python.
