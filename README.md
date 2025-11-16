# Project: Booking Intent Classification from Hotel Chatbot Conversations

**Overview:**
- Goal: Classify hotel chatbot conversations into booking intent stages:
  - low (exploring), medium (evaluating), high (actioning), abandoned (high but not completed in chat)
- What’s included:
  - Synthetic data generator (LLM-based, batched)
  - Modular intent classification framework
  - Baseline classifiers: rule-based, TF‑IDF + classical ML
  - Hugging Face model zero-shot pipeline
  - Evaluation scripts with standardized outputs

Requirements
- Python: managed by uv (supports 3.9–3.13 for this project)

**Environment Setup (uv)**
- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Create env: `uv venv` 
- Install all deps: `uv sync`
- Activate venv: `source .venv/bin/activate`
- Run scripts.

**Project Structure (brief overview)**
- data_gen/ (synthetic data generation code)
- intent_classification/
  - base/ (base classes, data structures, feature extractor)
  - data/ (loader, dataset stats)
  - evaluation/ (metrics, evaluator)
  - utils/ (logging, seeding, helpers)
  - classifiers/ (classifier implementations)
- evaluate_rule_classifier.py
- evaluate_ml_tfidf.py
- evaluate_hf_pipeline.py

**Data Format (JSONL)**
- One conversation per line
- Required fields: session_id, messages, intent_label
- messages is a list of {role: "user"|"bot", text: "..."}

Example:
```json
{"session_id":"abc123",
 "messages":[
   {"role":"user","text":"Hi, availability for 12–14 Oct for 2 adults?"},
   {"role":"bot","text":"Available: Deluxe €180/night. Proceed to book?"},
   {"role":"user","text":"Yes, book the Deluxe."}],
 "intent_label":"high"}
```
**How to use the repository:**

Set directory as src: `cd ./src`

**Generate Synthetic Data**
- Default (writes synthetic_conversations.jsonl):
```bash
python ./data_gen/synthetic_data_gen.py
```
- Custom count/output/model:
```bash
python ./data_gen/synthetic_data_gen.py \
  --count 500 \
  --output data/hotel_conversations.jsonl \
  --model "Qwen/Qwen3-1.7B"
```

**Evaluate Baselines and Models**
- All evaluators print the same summaries and save a results JSON.

Rule-based
```bash
python evaluate_rule_classifier.py
# Outputs: rule_classifier_results.json
```

TF‑IDF + Classical ML
```bash
python evaluate_ml_tfidf.py
# Outputs: ml_tfidf_results.json
```

HF Zero‑Shot (no training)
```bash
python evaluate_hf_pipeline.py
# Outputs: hf_pipeline_results.json
```

**How to Use Your Own Data**
- Put your JSONL at the src folder as synthetic_conversations.jsonl
  - or edit the dataset path inside the evaluation scripts
- Make sure intent_label is one of: low, medium, high, abandoned

**What the Evaluators Print/Save:**
- Printed:
  - Accuracy, Macro/Weighted Precision/Recall/F1
  - Per-class metrics
  - A few example predictions
  - Full classification report
- Saved:
  - {rule|ml_tfidf|hf_pipeline}_results.json with metrics and run summary


  ## TODO:
  - Rule based improvements: more keywords, better decision process;
  - Zero-shot speed-up testing script with just predict_proba;
  - Inference only script;