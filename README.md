# IGAP-DCLA

**IGAP-DCLA** is an inference-time framework for reducing hallucinations in Vision-Language Models (VLMs), particularly LLaVA-style architectures.

This repository implements a hybrid method that integrates:

- **Image-Guided Attention Pruning (IGAP)** — attention-level grounding
- **Dynamic Contrastive Logit Adjustment (DCLA)** — adaptive decoding control

The implementation also includes **SPIN** and **MoD** baselines for controlled comparison.

---

## Overview

Large Vision-Language Models often generate **hallucinated outputs**, producing objects or relations not present in the image. This behavior arises from over-reliance on language priors when visual grounding is weak.

IGAP-DCLA addresses this issue through:

- Attention-level intervention (IGAP)
- Logit-level adaptive correction (DCLA)
- Real-time hallucination detection via **JS divergence**

The method operates entirely at **inference time**, requiring no retraining.

---

## Architecture

The framework introduces a **dual-pass decoding mechanism** integrated into the autoregressive generation loop:

1. Standard forward pass → baseline logits  
2. IGAP forward pass → grounded logits  
3. JS divergence computation → hallucination risk  
4. DCLA routing → adaptive decoding decision  

### Architecture Diagram

<p align="center">
  <img src="assets/architecture.png" alt="IGAP-DCLA Architecture" width="85%">
</p>

The pipeline consists of:

- Vision encoder + projection module  
- LLM decoder backbone  
- Attention head pruning (IGAP)  
- Divergence-based routing  
- Dynamic logit adjustment (DCLA)  

---

## Method Components

### Image-Guided Attention Pruning (IGAP)

- Ranks attention heads based on visual token focus  
- Suppresses inattentive heads during decoding  
- Produces a grounded reference distribution without modifying image inputs  

### Hallucination Detection

- Computes **JS divergence** between:
  - Baseline logits
  - IGAP logits  
- High divergence indicates reliance on language priors  

### Dynamic Contrastive Logit Adjustment (DCLA)

Applies token-level routing:

| Condition | Action |
|----------|--------|
| High confidence | Greedy decoding |
| Low divergence | Complementary amplification |
| High divergence | Contrastive suppression + APC |

### Adaptive Plausibility Constraints (APC)

- Filters low-probability tokens
- Prevents unstable vocabulary shifts during contrastive decoding

---

## Repository Structure

```text
.
├── .vscode/
├── data/
│   ├── __init__.py
│   └── loader.py
├── eval/
│   ├── __init__.py
│   ├── generate_tables.py
│   ├── mmhal_inference.py
│   ├── mmhal_judge.py
│   └── pope_eval.py
├── model/
│   ├── __init__.py
│   ├── igap_dcla.py
│   ├── mod.py
│   └── spin.py
├── src/
│   ├── __init__.py
│   └── main.py
├── report.py
├── requirements.txt
└── README.md

```

## Installation

Create an environment and install the inference dependencies:

```bash
pip install -r requirements.txt
```

Required model checkpoint:

```text
llava-hf/llava-1.5-7b-hf
```

The MMHal-Bench archive is downloaded automatically to `data/test_data.zip` and extracted into `data/mmhal_data/` on first run.

## Running Inference

Run the hybrid method:

```bash
python src/main.py --benchmark igap_dcla
```

Other supported benchmarks:

```bash
python src/main.py --benchmark baseline
python src/main.py --benchmark spin
python src/main.py --benchmark mod
python src/main.py --benchmark all
```

Useful options:

```bash
python src/main.py --benchmark igap_dcla --limit 1
python src/main.py --benchmark igap_dcla --max-new-tokens 64
python src/main.py --benchmark igap_dcla --output-dir output/mmhal
```

Inference outputs are written to `output/mmhal/response_<benchmark>.json`.

## Reported Benchmark Results

The following values are the reported benchmark numbers for the method variants in this project.

### POPE

| Setting | Method | Acc | F1 |
| --- | --- | ---: | ---: |
| random | Baseline | 88.87 | 88.61 |
| random | SPIN | 87.00 | 86.87 |
| random | MoD | 86.00 | 86.41 |
| random | IGAP-DCLA | 89.09 | 89.87 |
| popular | Baseline | 86.40 | 86.52 |
| popular | SPIN | 87.00 | 86.87 |
| popular | MoD | 82.00 | 81.25 |
| popular | IGAP-DCLA | 88.00 | 87.76 |
| adversarial | Baseline | 76.00 | 78.18 |
| adversarial | SPIN | 76.00 | 78.18 |
| adversarial | MoD | 71.00 | 72.90 |
| adversarial | IGAP-DCLA | 79.00 | 80.37 |
| overall | Baseline | 84.09 | 84.77 |
| overall | SPIN | 83.33 | 83.97 |
| overall | MoD | 78.00 | 78.19 |
| overall | IGAP-DCLA | 84.67 | 85.00 |

### MMHal-Bench

| Category | Baseline | SPIN | MoD | IGAP-DCLA |
| --- | ---: | ---: | ---: | ---: |
| Comparison | 2.33 | 1.92 | 3.00 | 3.42 |
| Relation | 1.75 | 1.58 | 1.92 | 2.67 |
| Other | 1.92 | 1.67 | 1.83 | 2.75 |
| Holistic | 4.00 | 3.92 | 3.58 | 4.00 |
| Adversarial | 1.42 | 1.33 | 1.92 | 1.75 |
| Attribute | 3.25 | 2.08 | 1.83 | 3.00 |
| Environment | 3.42 | 2.17 | 3.33 | 2.75 |
| Counting | 2.25 | 2.00 | 2.33 | 1.92 |
| OVERALL | 2.54 | 2.08 | 2.47 | 2.78 |

## Optional Reporting

If you have MMHal and POPE evaluation JSON outputs, you can generate markdown summaries with:

```bash
python report.py
```

This writes markdown tables into `Report/` and reads inference outputs from `output/` with fallback support for legacy `eval/` and `pope_results/` directories.

## Notes

- This repository is intentionally kept inference-oriented; plot notebooks and plot-data scaffolding are not included.
- Legacy evaluation runners now live under `eval/`, and older helpers under `src/data`, `src/model`, and `src/decoding` remain as reference material.
- `README.md` is UTF-8 and the repository is normalized to LF line endings through `.gitattributes`.
