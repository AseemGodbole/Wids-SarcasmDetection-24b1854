# Sarcasm Detection with DistilBERT

This project implements a Natural Language Processing (NLP) pipeline to detect sarcasm in news headlines using a fine-tuned **DistilBERT** transformer model. The workflow moves from raw text data to model interpretability and deployment.

## Project Workflow

### 1. Environment & Data Exploration
We begin by setting up the environment with **Hugging Face Transformers**, **Pandas**, and **Scikit-Learn**.
- The dataset (`Sarcasm_Headlines_Dataset.json`) is loaded using Pandas.
- We perform Exploratory Data Analysis (EDA) to visualize the class balance between "Sarcastic" and "Not Sarcastic" headlines.

### 2. Tokenization & Data Preparation
Raw text is transformed into a format understandable by the model using `DistilBertTokenizerFast`.
- **Splitting:** Data is divided into Training (80%) and Validation (20%) sets using a stratified split to maintain class distribution.
- **Encoding:** Headlines are tokenized, truncated to a maximum length of 128 tokens, and padded.
- **Dataset:** A custom PyTorch `SarcasmDataset` class is created to handle batching.

### 3. Model Fine-Tuning
We load the pre-trained `distilbert-base-uncased` model and adapt it for binary classification.
- **Trainer API:** The Hugging Face `Trainer` handles the training loop, optimizing weights over 2 epochs.
- **Hyperparameters:** Includes a learning rate scheduler, warm-up steps, and weight decay to prevent overfitting.

### 4. Evaluation & Metrics
The model is evaluated on unseen validation data to ensure generalization.
- **Metrics:** We calculate Accuracy, Precision, Recall, and F1-Score.
- **Artifacts:** The final fine-tuned model and tokenizer are saved locally (`/sarcasm_model`) for future inference.

### 5. Interpretability & Deployment
We go beyond simple prediction to understand the model's logic.
- **LIME Analysis:** We use the LIME (Local Interpretable Model-agnostic Explanations) library to visualize which specific words trigger a "Sarcastic" prediction.
- **Streamlit App:** A reference script is provided to deploy the model as a simple web application where users can input headlines and receive real-time predictions.

---

## Installation

```bash
pip install transformers datasets scikit-learn pandas torch lime streamlit
