# AI-Based Video Metadata Generation Pipeline

This repository contains code and an example notebook for our end-to-end pipeline for automated, content-descriptive metadata generation in video archives. The approach integrates open-source AI models and supports flexible adaptation to organization-specific requirements.

---

## Features

* **Modular workflow** with six phases:

  1. Data acquisition
  2. Frame extraction
  3. Duplicate removal
  4. AI-based frame description (captioning, segmentation, object detection, multilabel classification, scene/night recognition, OCR/NER)
  5. Summarization to scene level
  6. Quantitative and qualitative evaluation

* **Open-source models only** for data control and cost-efficiency

* **Easily extensible:** Add models, adjust thresholds, and tailor outputs to your needs

---

## Quick Start

1. **Run the pipeline:**
   Open and execute `example_application.ipynb` for a demonstration, or use `Pipeline.py` directly for script-based runs.

2. **Customize:**
   Add your own videos and metadata, and adjust pipeline parameters within the notebook or script.

*Note: Please ensure all necessary Python packages are installed according to the import statements in the code.*

---

## Structure

```
Pipeline.py
example_application.ipynb
README.md
```
