# ds Expected Outputs

This folder stores the audio-level reference texts used for Phase 1 benchmarking.

## Files

- `manifest.csv`: master list of all audio files with their expected transcript, scenario, and utterance count.
- `per_audio/<audio_id>.csv`: one CSV per audio file, containing the same reference row for that audio.

## Matching Rule

Actual ASR outputs are matched against the expected text using:

- `audio_id`
- `filename`

The benchmark runner writes actual predictions to:

- `outputs/stt/phase_1_ds/<model>/predictions.csv`

It then joins the expected and actual rows on `audio_id` and `filename`.

## Metric Basis

All metrics are computed on cleaned text:

- punctuation removed
- whitespace normalized

Then the runner calculates:

- WER
- CER
- exact match accuracy
- latency per audio
- RTF = inference time / audio duration
