![](UTA-DataScience-Logo.png)

# The Echo Room — Dual-Lens Emotion Interpretation with LLMs

**A dual-emotion text classification tool that interprets both author intent and reader perception using fine-tuned transformer models.**

---

## Overview

Online messages are often emotionally misinterpreted. A post meant to express sadness might be read as sarcasm or anger. **The Echo Room** addresses this gap by analyzing a single piece of text from two emotional perspectives:
- The **author’s intended emotion**
- The **reader’s perceived reaction**

Using cleaned versions of the **GoEmotions** and **EmpatheticDialogues** datasets, we trained two separate models using Hugging Face Transformers. This project simulates digital empathy gaps and provides insights into tone divergence in social media discourse.

---

## Summary of Work

### Data

- **GoEmotions**:
  - Source: [Hugging Face](https://huggingface.co/datasets/go_emotions)
  - Type: Reddit comments (58K+)
  - Output: 27 fine-grained emotions + neutral
  - Used for **author emotion** prediction

- **EmpatheticDialogues**:
  - Source: [Hugging Face](https://huggingface.co/datasets/empathetic_dialogues)
  - Type: Dialog utterances labeled by emotion
  - Used for **reader emotion** prediction

- **Split**:
  - Train/Test ratio: 80/20 (stratified)
  - Label sets reduced to shared 10 emotion classes: `joy`, `anger`, `sadness`, `fear`, `surprise`, `disgust`, `love`, `gratitude`, `neutral`, `pride`

### Preprocessing / Cleanup

- Filtered each dataset to 10 shared emotion labels
- Mapped `EmpatheticDialogues` contexts to match GoEmotions label set
- Balanced label counts (e.g., capped 55k+ neutral labels in GoEmotions to 10k)
- Cleaned missing/invalid rows
- Applied `LabelEncoder` and `train_test_split`

### Data Visualization

- Bar plots for label distribution in both datasets
- Histogram of text lengths
- Confusion matrix and prediction distribution for test set

---

## Problem Formulation

- **Input**: A short piece of social media text (string)
- **Output**:
  - Label A: Emotion from the **author's perspective**
  - Label B: Emotion from a **typical reader's perspective**

### Model

- **Base Model**: [`BAAI/bge-small-en`](https://huggingface.co/BAAI/bge-small-en)
- **Architecture**: Transformer encoder with classification head
- **Loss**: Cross-entropy
- **Optimizer**: AdamW
- **Epochs**: 1
- **Batch Size**: 2 (gradient accumulation used to simulate larger batch)

---

## Training

- **Platform**: Google Colab (Free tier)
- **Train Time**: 1 epoch took **5 hours and 5 minutes**
- **Hardware Constraints**: Memory errors limited model size and training duration
- **Techniques Used**:
  - Gradient accumulation
  - Batch size reduction
  - Evaluation at each epoch
  - Model saving with `.zip` export

### Results (GoEmotions)

- **Train loss**: 1.0617
- **Eval loss**: 1.0211
- **Accuracy**: 66.75%
- **Precision**: 66.36%
- **F1 Score**: 66.33%

### Visuals

- Confusion matrix for 10 emotion classes
- Classification report
- Distribution of true vs predicted labels

---

## Conclusion

- The model can identify emotional divergence with ~66% accuracy.
- Author and reader emotions often differ, confirming the project's motivation.
- This type of model can be used for tone moderation, emotional education, or digital empathy analysis.

---

## Future Work

- Train reader model more explicitly using CARE dataset structure
- Extend to multi-label classification
- Deploy as a demo (e.g., Gradio or Streamlit app)
- Analyze divergence rates across demographic/post types

---

## How to Reproduce

1. Clone this repository.
2. Upload `go_emotions_dataset.csv` and `empatheticdialogues.tar.gz`.
3. Open and run `Echoroom.ipynb` in Google Colab.
4. Follow all cells in order to:
   - Clean datasets
   - Train the model
   - Test emotion prediction on sample text

---

## Files in Repository

- `Echoroom.ipynb`: Full implementation notebook (cleaning, training, evaluation)
- `Echoroom.pdf`: Static version for review
- `go_emotions_dataset.csv`: Cleaned Reddit-based emotion data (author)
- `empatheticdialogues.tar.gz`: Dialog dataset for reader emotion
- `LLM_Proposal_Zewdie.pdf`: Formal project proposal with goal, plan, and motivation
- `Image1.png` to `Image3.png`: Visuals (label plots, confusion matrix)
- `echo-room-model/`: Saved model and tokenizer
- `UTA-DataScience-Logo.png`: Branding

---

## Software Setup

- Python 3.10+
- Hugging Face Transformers (v4.37.2)
- PEFT, Accelerate
- PyTorch
- scikit-learn
- pandas, matplotlib

---

## Citations

- [GoEmotions Dataset – Hugging Face](https://huggingface.co/datasets/go_emotions)
- [EmpatheticDialogues Dataset – Hugging Face](https://huggingface.co/datasets/empathetic_dialogues)
- [BAAI/bge-small-en – Model Card](https://huggingface.co/BAAI/bge-small-en)
- Hugging Face Transformers Documentation
