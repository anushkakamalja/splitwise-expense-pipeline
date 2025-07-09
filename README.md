# 💸 Splitwise Expense Analysis Pipeline

An end-to-end **data engineering and analytics portfolio project** that processes, categorizes, and visualizes personal expenses using data exported from Splitwise.

> 🔧 Built with: Python · Airflow · ChromaDB · Sentence Transformers · Pandas · Streamlit · Snowflake (optional)

---

## 🚀 Project Overview

This project automates the entire workflow of:

1. **Ingesting Splitwise data**
2. **Anonymizing personally identifiable information (PII)**
3. **Embedding and categorizing expenses using a free vector database (ChromaDB)**
4. **Evaluating model accuracy**
5. **Visualizing insights in a dashboard**

---

## 📊 Key Features

- ✅ Expense categorization using embeddings (`all-MiniLM-L6-v2`)
- ✅ Customizable categories via editable CSV
- ✅ Confidence-based evaluation script
- ✅ Airflow pipeline for automation
- ✅ Streamlit dashboard to visualize trends
- ✅ Data anonymization for public sharing
- ✅ Modular codebase (easy to plug into Snowflake/BigQuery later)

---

## 🗂️ Project Structure

```bash
splitwise-expense-pipeline/
│
├── data/                       # Input and processed data
│   ├── expenses.csv
│   ├── expenses_anonymized.csv
│   ├── category_examples.csv
│   ├── test_labels.csv
│   └── eval_errors.csv
│
├── src/                        # All scripts
│   ├── fetch_splitwise_data.py
│   ├── anonymize_friends.py
│   ├── categorize_expenses.py
│   ├── evaluate_model.py
│   └── dashboard_app.py
│
├── dags/                       # Airflow DAGs
│   └── splitwise_etl_dag.py
│
├── .env                        # API keys and secrets
├── requirements.txt
└── README.md
