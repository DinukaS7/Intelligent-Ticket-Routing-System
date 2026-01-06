# Intelligent Ticket Routing System (SmartTriage)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791)
![Status](https://img.shields.io/badge/Status-Completed-green)

An end-to-end automated customer support triage system. This project ingests raw customer complaints, stores them in a **PostgreSQL** database, and uses a hybrid AI approach (**Random Forest** + **Deep Learning**) to automatically predict the **Priority** and **Department** for each ticket.

## 🚀 Project Overview

In high-volume customer support environments, manual triage is slow and expensive. This project automates that workflow using a full-stack data science pipeline:

1.  **Data Ingestion:** Loads raw text data from CSV into a structured SQL database.
2.  **Unsupervised Learning (LDA):** Automatically discovers support topics (e.g., "Billing", "Tech Support") from unlabeled text to create a labeled training set.
3.  **Priority Prediction (ML):** Uses **TF-IDF** and **Random Forest** to flag high-urgency tickets based on sentiment and keyword severity.
4.  **Department Routing (DL):** Uses a **TensorFlow/Keras** Neural Network (Embedding + GlobalAveragePooling) to route tickets to the correct department with **87%+ accuracy**.

## 🛠️ Tech Stack

* **Language:** Python
* **Database:** PostgreSQL (via `psycopg2` & `SQLAlchemy`)
* **Deep Learning:** TensorFlow / Keras
* **Machine Learning:** Scikit-Learn (Random Forest, LDA, TF-IDF)
* **NLP:** TextBlob (Sentiment), NLTK/Spacy (Preprocessing)
* **Visualization:** Matplotlib, Seaborn

## 📊 Project Architecture

The system follows a modular ETL and Inference pipeline:

```mermaid
graph LR
    A[Raw Data (CSV)] -->|Python Script| B[(PostgreSQL DB)]
    B -->|Fetch Text| C[Topic Modeling (LDA)]
    C -->|Update Labels| B
    B -->|Labeled Data| D{Model Training}
    D -->|TF-IDF + Random Forest| E[Priority Model]
    D -->|Tokenizer + Neural Net| F[Department Model]
    G[New Ticket] -->|Inference Engine| H[Final Report]
