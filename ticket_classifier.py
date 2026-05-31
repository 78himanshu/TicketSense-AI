"""

In my results, the few-shot classifier with the OpenAI vector store performed the best overall, achieving 0.72 accuracy and a weighted F1-score of 0.72. The few-shot classifier with the Gemma vector store was close behind at 0.69 accuracy, while both KNN models performed moderately and zero-shot performed the worst at 0.46 accuracy. Based on these results, I would choose the OpenAI embeddings with the few-shot classifier because it gave the strongest overall performance across most classes.

"""

import os
import shutil
from collections import Counter
from difflib import get_close_matches

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


TRAIN_FILE = "ticket_train.csv"
TEST_FILE = "ticket_test.csv"
LOCAL_GEMMA_PATH = "./embeddinggemma-300m"
OPENAI_CHROMA_DIR = "./chroma_openai"
GEMMA_CHROMA_DIR = "./chroma_gemma"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"
K_VALUE = 5
NUM_FEWSHOT_EXAMPLES = 3


def clean_old_store(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def make_documents(train_df: pl.DataFrame) -> list[Document]:
    documents = []
    for i, row in enumerate(train_df.iter_rows(named=True)):
        documents.append(
            Document(
                page_content=row["text"],
                metadata={"ticket_id": i, "label": row["label"]},
            )
        )
    return documents


def build_vector_store(documents: list[Document], embedding_function, persist_directory: str, collection_name: str):
    clean_old_store(persist_directory)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
    vector_store.add_documents(documents)
    return vector_store


def normalize_prediction(raw_prediction: str, categories: list[str]) -> str:
    cleaned = " ".join(raw_prediction.strip().split())
    for label in categories:
        if cleaned.lower() == label.lower():
            return label
    for label in categories:
        if label.lower() in cleaned.lower():
            return label
    closest = get_close_matches(cleaned, categories, n=1, cutoff=0.0)
    return closest[0] if closest else categories[0]


def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def zero_shot_classifier(client: OpenAI, ticket: str, categories: list[str]) -> str:
    system_prompt = f"""
You are a classifier machine that assigns categories to help desk tickets.
Respond with exactly one category from this list:
{", ".join(categories)}

Do not explain your answer.
Do not use quotes.
Do not add extra words.
""".strip()
    raw_prediction = call_llm(client, system_prompt, ticket)
    return normalize_prediction(raw_prediction, categories)


def knn_classifier(ticket: str, vector_store, k_value: int) -> str:
    neighbors = vector_store.similarity_search(ticket, k=k_value)
    labels = [doc.metadata["label"] for doc in neighbors]
    counts = Counter(labels)
    highest_count = max(counts.values())
    tied_labels = {label for label, count in counts.items() if count == highest_count}

    for label in labels:
        if label in tied_labels:
            return label

    return labels[0]


def few_shot_classifier(client: OpenAI, ticket: str, vector_store, categories: list[str], num_examples: int) -> str:
    retrieved_docs = vector_store.similarity_search(ticket, k=num_examples)

    example_blocks = []
    for i, doc in enumerate(retrieved_docs):
        example_blocks.append(
            f"""
<ticket id="example-{i}">
{doc.page_content}
</ticket>
<assistant_response id="example-{i}">
{doc.metadata["label"]}
</assistant_response>
""".strip()
        )

    examples_text = "\n\n".join(example_blocks)

    system_prompt = f"""
You are a classifier machine that assigns categories to help desk tickets.

Possible categories:
{", ".join(categories)}

Here are some labeled examples:
{examples_text}

Now classify the user's ticket.
Respond with exactly one category from the allowed list.
Do not explain your answer.
Do not use quotes.
Do not add extra words.
""".strip()

    raw_prediction = call_llm(client, system_prompt, ticket)
    return normalize_prediction(raw_prediction, categories)


def print_report(title: str, y_true: list[str], y_pred: list[str], categories: list[str]) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(classification_report(y_true, y_pred, labels=categories, zero_division=0))
    print()


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found. Put it in a .env file in this folder.")

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"{TRAIN_FILE} was not found in the current folder.")

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"{TEST_FILE} was not found in the current folder.")

    if not os.path.exists(LOCAL_GEMMA_PATH):
        raise FileNotFoundError(
            "The local EmbeddingGemma folder was not found at ./embeddinggemma-300m"
        )

    train_df = pl.read_csv(TRAIN_FILE)
    test_df = pl.read_csv(TEST_FILE)

    categories = sorted(train_df.get_column("label").unique().to_list())
    train_documents = make_documents(train_df)
    test_tickets = test_df.get_column("text").to_list()
    true_labels = test_df.get_column("label").to_list()

    print(f"Training rows: {train_df.height}")
    print(f"Test rows: {test_df.height}")
    print(f"Categories: {categories}")
    print()

    client = OpenAI()

    print("Loading embedding models...")
    openai_embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    gemma_embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_GEMMA_PATH,
        model_kwargs={"device": "cpu"},
    )

    print("Building OpenAI Chroma vector store...")
    openai_store = build_vector_store(
        documents=train_documents,
        embedding_function=openai_embeddings,
        persist_directory=OPENAI_CHROMA_DIR,
        collection_name="ticket_openai_store",
    )

    print("Building Gemma Chroma vector store...")
    gemma_store = build_vector_store(
        documents=train_documents,
        embedding_function=gemma_embeddings,
        persist_directory=GEMMA_CHROMA_DIR,
        collection_name="ticket_gemma_store",
    )

    print("Running zero-shot classifier...")
    zero_shot_predictions = [
        zero_shot_classifier(client, ticket, categories) for ticket in test_tickets
    ]

    print("Running KNN classifier with OpenAI vector store...")
    knn_openai_predictions = [
        knn_classifier(ticket, openai_store, K_VALUE) for ticket in test_tickets
    ]

    print("Running few-shot classifier with OpenAI vector store...")
    few_shot_openai_predictions = [
        few_shot_classifier(client, ticket, openai_store, categories, NUM_FEWSHOT_EXAMPLES)
        for ticket in test_tickets
    ]

    print("Running KNN classifier with Gemma vector store...")
    knn_gemma_predictions = [
        knn_classifier(ticket, gemma_store, K_VALUE) for ticket in test_tickets
    ]

    print("Running few-shot classifier with Gemma vector store...")
    few_shot_gemma_predictions = [
        few_shot_classifier(client, ticket, gemma_store, categories, NUM_FEWSHOT_EXAMPLES)
        for ticket in test_tickets
    ]

    print()
    print_report("ZERO-SHOT (no vector store)", true_labels, zero_shot_predictions, categories)
    print_report("KNN + OPENAI VECTOR STORE", true_labels, knn_openai_predictions, categories)
    print_report("FEW-SHOT + OPENAI VECTOR STORE", true_labels, few_shot_openai_predictions, categories)
    print_report("KNN + GEMMA VECTOR STORE", true_labels, knn_gemma_predictions, categories)
    print_report("FEW-SHOT + GEMMA VECTOR STORE", true_labels, few_shot_gemma_predictions, categories)

    print("Finished.")


if __name__ == "__main__":
    main()