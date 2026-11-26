%%writefile README.md
# Fine-Tuned Architectural Floor Layout Generator

An end-to-end pipeline to fine-tune a **Qwen-2.5-3B-Instruct** model using **Unsloth (LoRA)** to generate typical-floor architectural layouts in JSON format based on specific plot dimensions and zoning parameters.

## Project Overview
This project demonstrates how to bridge the gap between rule-based CAD systems and LLMs for generative design. 
1. **Rule Engine & Dataset Generator**: Generates thousands of valid, realistic building layouts based on architectural constraints (setbacks, footprint size, coverage ratio, core placement, corridor routes, and unit divisions).
2. **LoRA Fine-tuning**: Uses **Unsloth** to fine-tune the `Qwen2.5-3B-Instruct-bnb-4bit` model using instruction tuning, instructing the model to generate layout designs in JSON.
3. **Validation & Fallback**: Evaluates model generation with physical/geometrical validators, triggering self-correction cycles or falling back to the rule-engine if the model's design fails architectural constraints.

## How the Layout Engine Works
Given input parameters like:
- Plot Dimensions ($W \times D$ in meters)
- Setbacks (front, rear, side setbacks)
- Maximum footprint coverage (%)
- Number of Floors & Total Target Units

The engine calculates the allowable building envelope, places the core (elevators/stairs) and corridors, and partitions units with real-world target areas (Studio, 1-bed, 2-bed, etc.).

## Installation & Setup
Install the required tools for training:
```bash
pip install unsloth trl datasets matplotlib
```

## Dataset Generation
To generate 4,000 synthetic chat-format layout training dialogues:
```bash
python make_dataset.py --n 4000 --out data
```

## Fine-tuning (LoRA)
Fine-tune the model with low-rank adaptation using `FastLanguageModel` and `SFTTrainer` via Unsloth. Layout assistant outputs are trained using `train_on_responses_only` to focus optimization entirely on generating valid JSON layouts.

## Usage & Generation
Run the fine-tuned inference loop with visual matplotlib layout renderings:
```python
# Run layout generator
inp = {"W": 20, "D": 30, "sf": 3, "sr": 2, "ss": 1.5, "cov": 65, "floors": 4, "units": 16}
layout, source = design(inp)
print("Source:", source)
