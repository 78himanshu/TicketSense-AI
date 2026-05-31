# TicketSense-AI

A ticket classification system that compares OpenAI embeddings and local Gemma embeddings using LangChain, Chroma, and multiple classification strategies.

## Overview

TicketSense-AI builds two vector stores from customer support ticket data: one using OpenAI embeddings and another using the local `google/embeddinggemma-300m` embedding model. The system then evaluates zero-shot, KNN, and few-shot classification approaches on test tickets and prints classification reports to compare model performance.

The project demonstrates how vector databases and embedding models can be used for semantic ticket classification, routing, and support automation.

## What This Project Does

The pipeline:

1. Loads training and test ticket datasets
2. Builds a Chroma vector store with OpenAI embeddings
3. Builds a second Chroma vector store with local Gemma embeddings
4. Implements zero-shot ticket classification
5. Implements KNN-based classification using vector similarity search
6. Implements few-shot classification using retrieved examples
7. Evaluates classifier performance using `classification_report`

## Features

- Ticket classification using semantic embeddings
- Chroma vector store integration
- OpenAI embedding support
- Local Gemma embedding support
- Zero-shot classification
- KNN classification
- Few-shot classification
- Model comparison using classification metrics
- Lightweight CSV-based workflow

## Tech Stack

- Python
- LangChain
- ChromaDB
- OpenAI API
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- Polars
- NumPy

## Project Structure

```text
TicketSense-AI/
│
├── ticket_classifier.py
├── requirements.txt
├── ticket_train.csv
├── ticket_test.csv
├── model_results_1.png
├── model_results_2.png
├── .gitignore
└── README.md
```

Generated locally after running the script:

```text
chroma_openai/
chroma_gemma/
embeddinggemma-300m/
```

These generated folders are excluded from GitHub because they can be recreated locally.

## Installation

Clone the repository:

```bash
git clone https://github.com/78himanshu/TicketSense-AI.git
cd TicketSense-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control.

## Local Embedding Model Setup

This project expects the local Gemma embedding model to be available at:

```text
./embeddinggemma-300m
```

The model folder is not included in the repository because it is large and should be downloaded separately from Hugging Face.

## Usage

Run the script:

```bash
python ticket_classifier.py
```

The script will:

- Load the ticket datasets
- Build or load Chroma vector stores
- Run zero-shot, KNN, and few-shot classifiers
- Print classification reports for comparison

## Dataset

The project uses two CSV files:

```text
ticket_train.csv
ticket_test.csv
```

These contain labeled ticket examples used for training-style retrieval and evaluation.

## Example Output

The script prints classification metrics such as:

```text
precision
recall
f1-score
support
accuracy
macro avg
weighted avg
```

These metrics help compare the effectiveness of different embedding models and classification strategies.

## Why This Matters

Support ticket classification is a common real-world machine learning task. Automating ticket routing can reduce manual triage, improve response time, and help support teams prioritize issues more efficiently.

This project explores how embedding-based retrieval can be used as an alternative to traditional supervised classification, especially when labeled data is limited.

## Future Improvements

- Add a Streamlit dashboard for testing custom tickets
- Add confusion matrix visualizations
- Add support for more embedding models
- Store evaluation results in structured JSON
- Add batch prediction support
- Deploy the classifier as an API

## Author

Himanshu Paithane

