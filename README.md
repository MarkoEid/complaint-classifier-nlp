```markdown
# AI Customer Complaint Classification System

Automatically classifies financial customer complaints into their correct CFPB category using deep learning—comparing **SimpleRNN**, **LSTM**, and **GRU** (trained from scratch) against a fine-tuned **DistilBERT** transformer, deployed via an interactive Gradio dashboard.

---

## 📌 Overview

Financial institutions receive large volumes of free-text customer complaints that must be routed to the correct department (**credit reporting**, **debt collection**, **mortgages**, **credit cards**, **retail banking**). Manual triage is slow and inconsistent; this project fully automates that workflow.

---

## 📊 Dataset

Trained on the [Consumer Complaints Dataset for NLP](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp). Narratives are labeled across five categories:
* `credit_card`
* `credit_reporting`
* `retail_banking`
* `mortgages_and_loans`
* `debt_collection`

---

## ⚙️ Approach & Architecture

1. **Preprocessing:** Lowercasing, punctuation/number removal, stopword removal, and lemmatization.
2. **Class Imbalance:** Managed using computed class weights instead of synthetic oversampling.
3. **Model Pipeline:**
   * **SimpleRNN** (Baseline)
   * **LSTM** (Long Short-Term Memory)
   * **GRU** (Gated Recurrent Unit)
   * **DistilBERT** (Fine-tuned transformer via HuggingFace)
4. **Evaluation Metrics:** Accuracy, Macro Precision, Macro Recall, Macro F1, and Confusion Matrices.
5. **Interactive Deployment:** A Gradio dashboard featuring model selection, live confidence visualization, top-5 predictions, and execution stats.

---

## 📈 Performance Results

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **SimpleRNN** | 0.5652 | 0.3863 | 0.2470 | 0.2306 |
| **LSTM** | 0.8520 | 0.7958 | 0.8527 | 0.8206 |
| **GRU** | 0.8541 | 0.7977 | 0.8610 | 0.8255 |
| **DistilBERT (Transformer)** | **0.8972** | **0.8731** | **0.8584** | **0.8654** |

> **Best Model:** **Transformer (DistilBERT)** achieved the highest performance across overall accuracy, precision, and macro F1 score (0.8654).

---

## 🗂️ Repository Structure

| File / Directory | Purpose |
| :--- | :--- |
| `consumer_complaint_classification.ipynb` | Training notebook designed for **Kaggle GPU** execution. |
| `deployment_notebook.ipynb` | Loads training artifacts to serve the Gradio app directly on Kaggle. |
| `requirements.txt` | Pinned dependencies for local deployment  |

---

## 🚀 Getting Started

### 1. Training (Kaggle GPU)
1. Upload `consumer_complaint_classification.ipynb` to Kaggle.
2. Enable **GPU** and **Internet** in notebook settings.
3. Attach the dataset: **Add Data** ➔ search `Consumer Complaints Dataset for NLP`.
4. Run all cells to generate `saved_models/` and `artifacts/`.

### 2. Deployment Options

#### Option A: Run Locally (Recommended for Permanent App)
```bash
conda create -n complaint-app -y
conda activate complaint-app
pip install -r requirements.txt

```

Place the `saved_models/` and `artifacts/` directories in the root folder, then launch:

```bash
python app.py

```

*Access the app locally at: `http://127.0.0.1:7860*`

#### Option B: Deploy on Kaggle (Quick Temporary Demo)

1. Upload `deployment_notebook.ipynb` as a separate notebook.
2. Add your training notebook as an input source (**Add Input ➔ Notebook Output Files**).
3. Update `ARTIFACTS_ROOT` to your mounted path and run all cells to generate a public `gradio.live` link.

---

## ✨ Key Dashboard Features

* **Model Selector:** Switch between SimpleRNN, LSTM, GRU, and DistilBERT live.
* **Confidence Insights:** Visual confidence distribution bars and top-5 prediction tables.
* **Performance Telemetry:** Tracks per-model inference latency and input word counts.

---

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow / Keras, PyTorch, HuggingFace Transformers
* **NLP & Metrics:** NLTK, scikit-learn
* **Interface & Deployment:** Gradio

```

```
