# Step 4: Language Identification (LID)

This step identifies the language (Bengali, English, or mixed) of each token in the ASR transcript.

## 1. Benchmarking (Compare Models)
To run the full benchmark and see the metrics (WER/EN-WER) of both approaches:
```bash
# Run from the STT directory:
source venv/bin/activate
python3 run_experiment.py --step 4 --compare
```

## 2. Interactive Testing
To test specific sentences and see the token-level labels side-by-side:
```bash
# Run from the STT directory:
source venv/bin/activate
python3 test_lid.py "আপনার order টা কি confirm হয়েছে?"
```

## Approaches Implemented
- **4A: Rule-Based**: Baseline using Unicode ranges. (Reliable for Bengali suffixes).
- **4B: fastText**: Neural approach using Facebook's LID model. (Superior for English terms).
