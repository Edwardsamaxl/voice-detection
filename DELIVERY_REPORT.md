# Voice Detection Pipeline - Delivery Report

**Date:** 2026-05-19
**Evaluator:** Claude Code
**Dataset:** SLR80 Burmese ASR
**Pipeline Version:** v1.0 + Translation

---

## 1. Task Overview

This report evaluates the end-to-end voice detection pipeline on the SLR80 Burmese ASR dev set (100 utterances, 20 unique speakers). The pipeline consists of:

1. **Speaker Diarization** (pyannote.audio)
2. **Speaker Identification** (ECAPA-TDNN embeddings + FAISS vector index)
3. **ASR** (UniASR / FunASR)
4. **Translation** (NLLB-200 mya_Mymr -> zho_Hans)

The speaker repository (`speaker_db`) was built from the full training set embeddings using SLR80 filename-based speaker labels.

> **Note on Translation:** The evaluation script now runs the full pipeline **including translation** (`translator` is no longer hardcoded to `None`). Translation results are captured in the per-segment output JSON.

---

## 2. Experiment Setup

### 2.1 Training Set (Repository Build)

| Item | Value |
|------|-------|
| Source | `data/processed/burmese_asr/train/embeddings/` |
| Embedding Files | 4,660 NPZ files |
| Valid Segments | 2,528 segments with embeddings |
| Unique Speakers | 20 |
| Label Strategy | `slr80_filename` (extract speaker ID from filename) |
| Max Embeddings per Speaker | 20 |
| Output | `data/processed/speaker_db_full/` |

### 2.2 Validation Set (Evaluation)

| Item | Value |
|------|-------|
| Source | `data/raw/burmese_asr/dev/` |
| Audio Files | 100 WAV files |
| Unique Speakers | 20 (all present in training set) |
| Ground Truth | `speech_asr_slr80_my_devsets.csv` |
| Device | CPU |
| ASR Backend | UniASR (`models/uni_asr`) |
| Translation Model | NLLB-200-distilled-600M (`mya_Mymr` -> `zho_Hans`) |

### 2.3 Key Configuration

```python
CENTROID_THRESHOLD = 0.35
DISTANCE_THRESHOLD = 0.30
T_LOW  = 0.50   # identification rejection threshold
T_HIGH = 0.65   # high-confidence update threshold
MAX_EMB = 20
```

---

## 3. Speaker Identification Results

### 3.1 Overall Accuracy

| Metric | Value |
|--------|-------|
| Total Segments | 102 |
| Correct Speaker Matches | 96 |
| **Overall Speaker Accuracy** | **94.12%** |

### 3.2 Error Analysis

5 files (6 segments) had speaker misidentification:

| File | True Speaker | Correct / Total |
|------|-------------|-----------------|
| `bur_5189_0145018221.wav` | 5189 | 1/2 |
| `bur_5362_5245492029.wav` | 5362 | 0/1 |
| `bur_5362_6868203464.wav` | 5362 | 0/2 |
| `bur_6118_8081481703.wav` | 6118 | 0/1 |
| `bur_8266_7258038860.wav` | 8266 | 0/1 |

Speaker **5362** was the most problematic (3 errors across 11 files), suggesting this speaker's utterances may have acoustic similarity with other speakers or lower audio quality.

### 3.3 Per-Speaker Breakdown

| Speaker | Accuracy | CER | Files |
|---------|----------|-----|-------|
| 4632 | 100% | 0.5714 | 5 |
| 6118 | 85.71% | 0.5864 | 7 |
| 4409 | 100% | 0.6031 | 4 |
| 0366 | 100% | 0.6032 | 3 |
| 5903 | 100% | 0.6035 | 9 |
| 8266 | 80% | 0.6041 | 5 |
| 0644 | 100% | 0.6106 | 2 |
| 5932 | 100% | 0.6119 | 10 |
| 5189 | 85.71% | 0.6200 | 7 |
| 2446 | 100% | 0.6261 | 6 |
| 6884 | 100% | 0.6324 | 2 |
| 7712 | 100% | 0.6404 | 9 |
| 9135 | 100% | 0.6431 | 4 |
| 9762 | 100% | 0.6448 | 6 |
| 3260 | 100% | 0.6557 | 4 |
| 8698 | 100% | 0.6618 | 3 |
| 5362 | 72.73% | 0.6643 | 11 |
| 7865 | 100% | 0.6722 | 3 |
| 7543 | 100% | 0.7027 | 2 |

**Observation:** Speaker identification accuracy is consistently high across most speakers. The few errors are likely due to borderline similarity scores falling below `T_HIGH = 0.65`, causing the system to assign a new speaker ID instead of matching the known speaker.

---

## 4. ASR Results

### 4.1 Overall Metrics

| Metric | Value |
|--------|-------|
| Overall CER | **0.6250** (62.50%) |
| Overall Precision | **0.3750** (37.50%) |
| Mean Precision (per file) | 37.28% |
| Median Precision | 37.42% |
| Min Precision | 15.28% |
| Max Precision | 58.43% |
| Std Dev | 7.26% |

### 4.2 ASR Quality Assessment

The ASR precision of **37.5%** indicates significant room for improvement. Character Error Rate (CER) of 62.5% means that, on average, more than half of the characters in the reference transcript are incorrect in the hypothesis.

Example of ASR output vs. reference:

**Reference:**
> ဒီ ကိစ္စ နဲ့ ပတ်သက်လို့ ဒေသခံ တစ်ဦး ဖြစ်တဲ့ ကိုဖြိုးဟန် က အခုလို ပြောပါတယ် ဒေါ်လာ

**Hypothesis:**
> ဒီကိစ္စ နဲ့လို့ခံတစ်ဦးတဲ့ကိုန်ပတ်သက်ဒေသဖြစ်ဖြိုးဟက လိုပါတယ်လာ

**Issues observed:**
- Missing spaces between words (word boundary errors)
- Character substitution errors (e.g., `ဟန်` -> `န်ပ်`, `ပြော` -> `လို`)
- Missing terminal `ဒေါ်လာ` marker in some cases
- Overall the model captures the general phonetic structure but misses many fine-grained characters

### 4.3 Per-Speaker ASR Variance

CER varies moderately across speakers (range: 0.57 - 0.70), but the difference is not dramatic. This suggests the ASR model's poor performance is **not speaker-specific** but rather a **global model limitation** for the Burmese language on this dataset.

---

## 5. Translation Results

### 5.1 Pipeline Integration Status

| Item | Status |
|------|--------|
| Translation module loaded | Yes |
| Translation executed per segment | Yes |
| Output JSON contains `translation` field | Yes |
| Translation time overhead (CPU) | ~+1-2s per file |

Translation is now a **first-class citizen** in the evaluation pipeline. The `eval_pipeline_on_dev.py` script loads `Translator` by default (pass `--no_translate` to skip).

### 5.2 Translation Quality Samples

The following samples illustrate the **current translation quality** when fed with UniASR output (CER ~60%):

| File | ASR Hypothesis (Myanmar) | Translation (Chinese) | Assessment |
|------|-------------------------|----------------------|------------|
| `bur_0366_5281755035.wav` | `ဒီကိစ္စ နဲ့လို့ခံတစ်ဦးတဲ့ကိုန်ပတ်သက်ဒေသဖြစ်ဖြိုးဟက လိုပါတယ်လာ` | 需要一个特定的领域来解决这个问题. | Roughly understandable; context partially preserved |
| `bur_0644_0716488921.wav` | `ကျမ ဆောင်နဲ့ကားရဲ့သူသရုပ်ရုပ်ရှင်ထုတ်လုပ်က စ်ပါလာ` | 我和我的女主角在电影中拍摄了电影. | Semantically coherent despite ASR errors |
| `bur_0366_7193794246.wav` | `ောကိုမှုဖြင့်ောင်ထဲသူိမ်ံဖြစ်သည်လာ` | 子子子子子子子子子子子子... | **Complete failure**: ASR output is garbled, causing NLLB to hallucinate repetitive tokens |

### 5.3 Translation Quality Assessment

**Root cause analysis:**

The translation model (NLLB-200 600M) itself is **not the primary bottleneck**. The dominant issue is **ASR quality feeding into translation**:

- When ASR produces structurally reasonable output (even with substitution errors), NLLB can often produce **semantically intelligible** Chinese.
- When ASR collapses into garbled sequences (as in sample 3 above), NLLB has no meaningful source to translate and produces **repetitive nonsense**.

This is a classic **garbage-in, garbage-out** scenario.

**Translation model suitability:**

For Burmese -> Chinese, NLLB-200 600M is an acceptable baseline for an offline, self-hosted system. Higher-quality alternatives exist (NLLB-200 3.3B, MiLMMT-46, Sailor2-20B), but upgrading the translation model without first improving ASR will yield **diminishing returns**.

---

## 6. Performance & Timing

| Metric | Value (No Translation) | Value (With Translation) |
|--------|------------------------|--------------------------|
| Average time per file | ~3.5s - 4.5s | ~5s - 6s |
| Total evaluation time (100 files, CPU) | ~12 minutes | ~18-20 minutes |
| Bottleneck | UniASR model initialization per file | UniASR init + NLLB decode |

**Note:** The ASR model is currently re-initialized for every file. Reusing a single model instance across files would significantly reduce total evaluation time.

---

## 7. Conclusions

### 7.1 What Works Well

1. **Speaker Identification (94.12% accuracy):** The ECAPA-TDNN embeddings combined with the FAISS-based two-stage identification (center coarse ranking + per-speaker embedding averaging) performs very well. Using SLR80 filename labels provides a clean, deterministic speaker mapping.
2. **Diarization:** pyannote.audio reliably segments single-speaker utterances into 1-2 segments per file.
3. **Pipeline Stability:** The entire pipeline ran through all 100 files without crashes.
4. **Translation Integration:** The NLLB-200 translation module is fully wired into the pipeline and produces output for every segment.

### 7.2 Known Limitations (Delivery Notes)

The following issues are **acknowledged and documented** for the current delivery:

| Component | Issue | Severity | Impact on Delivery |
|-----------|-------|----------|-------------------|
| **ASR** | CER = 62.5%, precision = 37.5% | **High** | Transcripts are largely incorrect; this is the single biggest quality issue |
| **Translation** | Quality severely degraded by ASR errors | **High** | Understandable when ASR is okay; collapses when ASR is garbled |
| **Model Efficiency** | UniASR reloaded per file | Medium | Wastes ~1-2s per file on initialization |

**Bottom line for delivery:**

- The **speaker identification subsystem is production-ready** (94.12% accuracy, stable, fast).
- The **ASR subsystem is the critical blocker**. It runs correctly from an engineering standpoint (no crashes, outputs text), but the **linguistic quality is not production-ready** for Burmese.
- The **translation subsystem is correctly integrated** and will improve automatically as ASR quality improves.

---

## 8. Recommendations

### Short Term (Post-Delivery Priority)

1. **Fix ASR Precision:**
   - The ASR model (`models/uni_asr`) is the critical weakness. Consider:
     - Fine-tuning on the SLR80 training set
     - Using a larger/fine-tuned model (e.g., XLS-R CTC or Whisper large-v3)
     - The project already has `scripts/train_xlsr_ctc.py` and `scripts/eval_xlsr_ctc.py` -- training a custom CTC model on SLR80 may yield better results
   - Add text normalization specifically for Burmese (the current `src/asr/text_normalize.py` handles basic normalization but the ASR model itself needs improvement)

2. **Optimize Pipeline Speed:**
   - Cache the UniASR pipeline instance in `src/asr/uniasr_asr.py` to avoid reloading the model for every audio file
   - Similar caching can be applied to the pyannote diarization pipeline

### Medium Term

1. **Upgrade Translation Model (only after ASR is fixed):**
   - If ASR CER drops below ~30%, consider upgrading from NLLB-600M to **NLLB-200-3.3B** for better translation fluency
   - For maximum quality, evaluate **MiLMMT-46** or **Sailor2-20B** (Southeast-Asia specialized)

2. **Tune Identification Thresholds:**
   - `T_LOW = 0.50` and `T_HIGH = 0.65` work well overall, but analyzing the score distribution for the 5 error cases could reveal if adjusting these thresholds would help

3. **Cross-Speaker Analysis:**
   - Investigate why speaker 5362 has the highest error rate. It may have more acoustic variability or shorter utterances

---

## 9. Artifacts

| Artifact | Path |
|----------|------|
| Speaker DB | `data/processed/speaker_db_full/` |
| Evaluation Script (with translation) | `scripts/eval_pipeline_on_dev.py` |
| Detailed Results | `data/processed/eval_results/eval_report.json` |

---

## 10. Summary

| Component | Status | Score | Notes |
|-----------|--------|-------|-------|
| Speaker Diarization | OK | - | Stable, segments correctly |
| Speaker Identification | **Good** | **94.12%** | Production-ready |
| ASR Precision | **Poor** | **37.50%** | Engineering-complete, quality-blocked |
| Translation Integration | OK | - | Wired and running; quality tied to ASR |
| Pipeline Stability | OK | 100/100 files processed | No crashes |

**Delivery Statement:**

The voice detection pipeline is **functionally complete and stable**. All four stages (diarization, identification, ASR, translation) execute end-to-end without errors. The speaker identification subsystem meets production quality. The ASR subsystem is the acknowledged weak link and should be the top priority for the next iteration.
