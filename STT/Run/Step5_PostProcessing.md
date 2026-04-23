# Step 5: Post-Processing

This step performs transcript cleanup after ASR inference (spelling correction, dialect normalization, punctuation restoration, selective LLM cleanup).

## 1. Benchmarking (Compare Approaches)

To compare all logged Step 5 runs by WER:

```bash
# Run from the STT directory:
source venv/bin/activate
python3 run_experiment.py --step 5 --compare
```

## 2. Run Individual Approaches

```bash
# Run from the STT directory:
source venv/bin/activate
python3 run_experiment.py --approach 5A
python3 run_experiment.py --approach 5B
python3 run_experiment.py --approach 5C
python3 run_experiment.py --approach 5D
python3 run_experiment.py --approach 5E
python3 run_experiment.py --approach 5F
```

## Approaches Implemented

- **5A: English Word Correction**: Converts common Bengali-script English variants (e.g., অর্ডার) to English forms (order).
- **5B: Dialectal Normalization**: Normalizes frequent colloquial/dialect forms (e.g., হইছে -> হয়েছে).
- **5C: Rule-Based Punctuation**: Adds sentence-ending punctuation with simple question-cue detection.
- **5D: ML Punctuation**: Uses `deepmultilingualpunctuation` when available, otherwise falls back to 5C.
- **5E: LLM Post-Processing (Selective)**: Uses `gpt-4o-mini` only for low-quality transcripts; falls back to 5B+5C if unavailable.
- **5F: Full Stack**: Applies 5A+5B+5C and then selective 5E.

## Optional Dependencies

- For **5D**:
  - `pip install deepmultilingualpunctuation`
- For **5E/5F**:
  - `pip install openai`
  - Set `OPENAI_API_KEY` in the environment
  - Optional model override: `export STT_POSTPROC_LLM_MODEL=gpt-4o-mini`
