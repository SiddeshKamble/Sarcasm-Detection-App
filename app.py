import os
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import joblib
import seaborn as sns  # kept because you import it
from collections import Counter
from wordcloud import WordCloud

import requests
import re

# Detect Streamlit Cloud (used to keep hosted demo stable)
ON_STREAMLIT = bool(os.getenv("STREAMLIT_CLOUD")) or bool(os.getenv("STREAMLIT_SERVER_PORT")) or bool(os.getenv("STREAMLIT_SERVER_RUNNING"))

# Only import torch/transformers locally (prevents Streamlit Cloud crash)
if not ON_STREAMLIT:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


DRIVE_FILE_ID = "1BQaeAZfiMXHKwAZ9yEyO1VH5wTPyskIk"
LOCAL_PATH = "sarcasm.csv"


def download_from_drive(file_id: str, dest: str):
    """
    Downloads large Google Drive files by handling the 'virus scan warning' confirm token.
    """
    URL = "https://drive.usercontent.google.com/download"
    session = requests.Session()

    # First request (may return the warning HTML)
    r = session.get(URL, params={"id": file_id}, stream=True)
    r.raise_for_status()

    # If we got HTML, extract confirm token and retry
    content_type = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type:
        html = r.text

        # Try to find confirm token from hidden input: name="confirm" value="t" or similar
        m = re.search(r'name="confirm"\s+value="([^"]+)"', html)
        confirm = m.group(1) if m else "t"

        # Retry with confirm token
        r = session.get(URL, params={"id": file_id, "confirm": confirm}, stream=True)
        r.raise_for_status()

    # Write file to disk
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


@st.cache_data
def load_data():
    # Download once
    if not os.path.exists(LOCAL_PATH) or os.path.getsize(LOCAL_PATH) < 1024:
        with st.spinner("Downloading dataset..."):
            download_from_drive(DRIVE_FILE_ID, LOCAL_PATH)

    df = pd.read_csv(LOCAL_PATH)

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure expected column exists
    if "comment" not in df.columns:
        st.error(f"CSV columns found: {df.columns.tolist()}")
        raise KeyError("Missing 'comment' column in dataset")

    df = df.dropna(subset=["comment"])
    df["comment_length"] = df["comment"].astype(str).str.len()
    return df


@st.cache_resource
def train_all_models(df: pd.DataFrame):
    """
    Trains NB + LR always.
    Trains transformer models only when NOT on Streamlit Cloud (to keep hosted demo stable).
    """
    metrics = {}
    try:
        # Naive Bayes
        vectorizer_nb = CountVectorizer()
        X_nb = vectorizer_nb.fit_transform(df["comment"].astype(str))
        y_nb = df["label"]

        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
            X_nb, y_nb, test_size=0.25, random_state=42
        )

        model_nb = MultinomialNB()
        model_nb.fit(X_train_nb, y_train_nb)
        y_pred_nb = model_nb.predict(X_test_nb)
        metrics["Naive Bayes"] = classification_report(
            y_test_nb, y_pred_nb, output_dict=True
        )
        joblib.dump((model_nb, vectorizer_nb), "naive_bayes_model.pkl")

        # Logistic Regression
        vectorizer_lr = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_lr = vectorizer_lr.fit_transform(df["comment"].astype(str))
        y_lr = df["label"]

        X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
            X_lr, y_lr, test_size=0.25, random_state=42
        )

        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train_lr, y_train_lr)
        y_pred_lr = model_lr.predict(X_test_lr)
        metrics["Logistic Regression"] = classification_report(
            y_test_lr, y_pred_lr, output_dict=True
        )
        joblib.dump((model_lr, vectorizer_lr), "logistic_regression_model.pkl")

        # Transformers: disable on Streamlit Cloud
        if ON_STREAMLIT:
            metrics["Pretrained Transformer"] = {
                "note": "Disabled on Streamlit Cloud to keep the app stable."
            }
            metrics["DistilBERT Model"] = {
                "note": "Disabled on Streamlit Cloud to keep the app stable."
            }
            model_transformer, tokenizer = None, None
            distilbert_model, distilbert_tokenizer = None, None
        else:
            # Pretrained Transformer Model (RoBERTa)
            tokenizer = AutoTokenizer.from_pretrained(
                "jkhan447/sarcasm-detection-RoBerta-base-CR"
            )
            model_transformer = AutoModelForSequenceClassification.from_pretrained(
                "jkhan447/sarcasm-detection-RoBerta-base-CR"
            )

            # Use a smaller subset of data for evaluation
            df_subset = df.sample(n=min(500, len(df)), random_state=42)
            y_test_transformer = df_subset["label"].tolist()
            y_pred_transformer = []

            batch_size = 16
            for i in range(0, len(df_subset), batch_size):
                batch_comments = df_subset["comment"].iloc[i : i + batch_size].astype(str).tolist()
                tokenized_texts = tokenizer(
                    batch_comments,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    output = model_transformer(**tokenized_texts)
                probs = torch.softmax(output.logits, dim=-1).tolist()
                y_pred_transformer.extend([1 if prob[1] > 0.5 else 0 for prob in probs])

            metrics["Pretrained Transformer"] = classification_report(
                y_test_transformer, y_pred_transformer, output_dict=True
            )
            joblib.dump((model_transformer, tokenizer), "pretrained_transformer_model.pkl")

            # DistilBERT Model
            distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            distilbert_model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased"
            )

            y_test_distilbert = df_subset["label"].tolist()
            y_pred_distilbert = []

            for i in range(0, len(df_subset), batch_size):
                batch_comments = df_subset["comment"].iloc[i : i + batch_size].astype(str).tolist()
                tokenized_texts = distilbert_tokenizer(
                    batch_comments,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    output = distilbert_model(**tokenized_texts)
                probs = torch.softmax(output.logits, dim=-1).tolist()
                y_pred_distilbert.extend([1 if prob[1] > 0.5 else 0 for prob in probs])

            metrics["DistilBERT Model"] = classification_report(
                y_test_distilbert, y_pred_distilbert, output_dict=True
            )
            joblib.dump((distilbert_model, distilbert_tokenizer), "distilbert_model.pkl")

        return (
            (model_nb, vectorizer_nb),
            (model_lr, vectorizer_lr),
            (model_transformer, tokenizer),
            (distilbert_model, distilbert_tokenizer),
            metrics,
        )

    except Exception as e:
        st.error(f"An error occurred while training models: {e}")
        return None, None, None, None, None


def predict_sarcasm_transformer(text: str, tokenizer, model):
    tokenized_text = tokenizer(
        [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**tokenized_text)
    probs = torch.softmax(output.logits, dim=-1).tolist()[0]
    return probs[1]  # Probability of sarcasm


def predict_sarcasm_distilbert(text: str, tokenizer, model):
    tokenized_text = tokenizer(
        [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**tokenized_text)
    probs = torch.softmax(output.logits, dim=-1).tolist()[0]
    return probs[1]  # Probability of sarcasm


def display_metrics(metrics_dict):
    for model_name, metrics in metrics_dict.items():
        st.write(f"### {model_name} Metrics:")
        if isinstance(metrics, dict):
            metrics_df = pd.DataFrame(metrics).transpose()
            st.write(metrics_df)
        else:
            st.write(metrics)


def main():
    st.set_page_config(layout="wide")
    st.title("Sarcasm Detection App 👾💬")

    st.sidebar.title("Navigation")
    navigation = st.sidebar.radio(
        "Choose a section:",
        [
            "🏠 Home",
            "📊 Dataset Overview",
            "📈 EDA",
            "🛠️ Model Training",
            "🤖 Interactive Prediction",
        ],
    )

    df = load_data()

    if navigation == "🏠 Home":
        st.header("Welcome to the Sarcasm Detection App!")
        st.markdown(
            """
            ### 🤔 What is Sarcasm Detection?
            This application uses **machine learning** and **deep learning** models to detect sarcasm in text. 🌐
            
            💡 **Features:**
            - Explore the dataset 📊
            - View interesting visualizations 📈
            - Train models to detect sarcasm 🛠️
            - Test sarcasm predictions interactively 🤖
            
            > "Sarcasm is the lowest form of wit but the highest form of intelligence." – Oscar Wilde 🦅

            **Have fun exploring sarcasm detection!**
            """
        )

        if ON_STREAMLIT:
            st.info(
                "Note: Transformer training is disabled on the hosted demo to keep it fast and stable. "
                "Naive Bayes and Logistic Regression work normally."
            )

    elif navigation == "📊 Dataset Overview":
        st.header("Dataset Overview 🛂️")
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        st.write("Class Distribution:")
        st.bar_chart(df["label"].value_counts())

    elif navigation == "📈 EDA":
        st.header("Exploratory Data Analysis (EDA) 📈")

        columns = ["label", "comment", "subreddit", "score", "parent_comment"]
        data_clean = df[columns].copy()

        # Sarcasm Proportion by Length Bins
        data_clean["comment_length"] = data_clean["comment"].astype(str).str.len()
        data_clean["length_bin"] = pd.cut(
            data_clean["comment_length"],
            bins=[0, 50, 100, 200, 300, 500, np.inf],
            labels=["0-50", "51-100", "101-200", "201-300", "301-500", "500+"],
        )

        sarcasm_by_length = data_clean.groupby("length_bin", observed=False)["label"].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        sarcasm_by_length.plot(kind="bar", color="green", alpha=0.7, ax=ax)
        ax.set_title("Sarcasm Proportion by Length Bins")
        ax.set_xlabel("Length Bin")
        ax.set_ylabel("Sarcasm Proportion")
        plt.tight_layout()
        st.pyplot(fig)

        # Tokenize and count words
        sarcastic_words = " ".join(
            data_clean[data_clean["label"] == 1]["comment"].dropna().astype(str)
        ).split()
        non_sarcastic_words = " ".join(
            data_clean[data_clean["label"] == 0]["comment"].dropna().astype(str)
        ).split()

        sarcastic_counter = Counter(sarcastic_words).most_common(20)
        non_sarcastic_counter = Counter(non_sarcastic_words).most_common(20)

        sarcastic_df = pd.DataFrame(sarcastic_counter, columns=["Word", "Count"])
        non_sarcastic_df = pd.DataFrame(non_sarcastic_counter, columns=["Word", "Count"])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            sarcastic_df["Word"],
            sarcastic_df["Count"],
            alpha=0.7,
            label="Sarcasm",
            color="blue",
        )
        ax.bar(
            non_sarcastic_df["Word"],
            non_sarcastic_df["Count"],
            alpha=0.7,
            label="Not Sarcasm",
            color="orange",
        )
        ax.set_title("Top Words in Sarcastic vs. Non-Sarcastic Comments")
        words = sarcastic_df["Word"].tolist()
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Top Subreddits with Most Sarcastic Comments
        if "subreddit" in data_clean.columns:
            top_subreddits = df[df["label"] == 1]["subreddit"].value_counts().head(10)

            fig, ax = plt.subplots(figsize=(8, 4))
            top_subreddits.plot(kind="bar", color="red", alpha=0.8, ax=ax)
            ax.set_title("Top 10 Subreddits with Most Sarcastic Comments")
            ax.set_xlabel("Subreddit")
            ax.set_ylabel("Count of Sarcastic Comments")
            ax.set_xticklabels(top_subreddits.index, rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        # Distribution of Reddit Scores
        if "score" in data_clean.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(
                data_clean[data_clean["label"] == 1]["score"].clip(-10, 50),
                bins=50,
                alpha=0.7,
                label="Sarcasm",
                color="blue",
            )
            ax.hist(
                data_clean[data_clean["label"] == 0]["score"].clip(-10, 50),
                bins=50,
                alpha=0.7,
                label="Not Sarcasm",
                color="orange",
            )
            ax.set_title("Distribution of Reddit Scores")
            ax.set_xlabel("Score (clipped at -10 and 50)")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        # Word Cloud of Sarcastic Comments
        sarcastic_text = " ".join(
            data_clean[data_clean["label"] == 1]["comment"].dropna().astype(str)
        )
        wordcloud_sarcasm = WordCloud(width=800, height=400, background_color="white").generate(
            sarcastic_text
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud_sarcasm, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud of Sarcastic Comments")
        plt.tight_layout()
        st.pyplot(fig)

        # Word Cloud of Non-Sarcastic Comments
        non_sarcastic_text = " ".join(
            data_clean[data_clean["label"] == 0]["comment"].dropna().astype(str)
        )
        wordcloud_non_sarcasm = WordCloud(
            width=800, height=400, background_color="white"
        ).generate(non_sarcastic_text)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud_non_sarcasm, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud of Non-Sarcastic Comments")
        plt.tight_layout()
        st.pyplot(fig)

    elif navigation == "🛠️ Model Training":
        st.header("Model Training 🏃️‍♂️")

        if (
            "nb_model" in st.session_state
            and "lr_model" in st.session_state
            and "pretrained_model" in st.session_state
            and "distilbert_model" in st.session_state
        ):
            st.write("Models are already trained. You can proceed to Interactive Prediction. 🤖")
            if st.button("Clear Session State and Retrain Models 🔄"):
                for key in [
                    "nb_model",
                    "lr_model",
                    "pretrained_model",
                    "distilbert_model",
                    "metrics",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            if st.button("Train All Models 🚀"):
                (
                    nb_model_data,
                    lr_model_data,
                    transformer_model_data,
                    distilbert_model_data,
                    metrics,
                ) = train_all_models(df)

                if metrics:
                    st.session_state["nb_model"] = nb_model_data
                    st.session_state["lr_model"] = lr_model_data
                    st.session_state["pretrained_model"] = transformer_model_data
                    st.session_state["distilbert_model"] = distilbert_model_data
                    st.session_state["metrics"] = metrics
                    st.success("Training complete. Go to Interactive Prediction. ✅")

    elif navigation == "🤖 Interactive Prediction":
        st.header("Interactive Prediction 🎯")
        st.write("Example Sentences for Testing:")
        st.markdown(" - Thank you for your feedback. It WaS ReAlLy InSIgGhTFul!")
        st.markdown(" - Wow, I really didn’t expect you to pass that exam. Good for you!!!")
        st.markdown(" - Methodology: Crafting the Ultimate Seriousness Detector")

        user_input = st.text_input("Enter a sentence to detect sarcasm 📝:")

        if (
            "nb_model" not in st.session_state
            or "lr_model" not in st.session_state
            or "pretrained_model" not in st.session_state
            or "distilbert_model" not in st.session_state
        ):
            st.write("Please train all models first! 🚧")
        elif user_input:
            nb_model, nb_vectorizer = st.session_state["nb_model"]
            nb_prediction = nb_model.predict(nb_vectorizer.transform([user_input]))[0]

            lr_model, lr_vectorizer = st.session_state["lr_model"]
            lr_prediction = lr_model.predict(lr_vectorizer.transform([user_input]))[0]

            pretrained_model, tokenizer = st.session_state["pretrained_model"]
            distilbert_model, distilbert_tokenizer = st.session_state["distilbert_model"]

            # If running on Streamlit Cloud, these will be None
            if (
                pretrained_model is None
                or tokenizer is None
                or distilbert_model is None
                or distilbert_tokenizer is None
            ):
                st.info(
                    "Transformer models are disabled on the hosted demo to keep it fast and stable. "
                    "Naive Bayes and Logistic Regression predictions are shown below."
                )
                st.write(
                    "Naive Bayes Prediction:",
                    "😏 Sarcasm" if nb_prediction else "🙂 Not Sarcasm",
                )
                st.write(
                    "Logistic Regression Prediction:",
                    "😏 Sarcasm" if lr_prediction else "🙂 Not Sarcasm",
                )
            else:
                pretrained_sarcasm = predict_sarcasm_transformer(
                    user_input, tokenizer, pretrained_model
                )
                pretrained_prediction = (
                    "Sarcasm detected" if pretrained_sarcasm > 0.5 else "No sarcasm detected"
                )

                distilbert_sarcasm = predict_sarcasm_distilbert(
                    user_input, distilbert_tokenizer, distilbert_model
                )
                distilbert_prediction = (
                    "Sarcasm detected" if distilbert_sarcasm > 0.5 else "No sarcasm detected"
                )

                st.write(
                    "Naive Bayes Prediction:",
                    "😏 Sarcasm" if nb_prediction else "🙂 Not Sarcasm",
                )
                st.write(
                    "Logistic Regression Prediction:",
                    "😏 Sarcasm" if lr_prediction else "🙂 Not Sarcasm",
                )
                st.write(f"Pretrained Model Prediction: {pretrained_prediction}")
                st.write(f"DistilBERT Model Prediction: {distilbert_prediction}")

    # Sidebar: metrics display
    if "metrics" in st.session_state and st.session_state["metrics"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Metrics (latest)")
        display_metrics(st.session_state["metrics"])


if __name__ == "__main__":
    main()