import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# -----------------------------
# 📥 Step 1: Load anonymized expenses
# -----------------------------
print("📥 Loading expense data...")
df = pd.read_csv("data/expenses_anonymized.csv")
descriptions = df["description"].fillna("").astype(str).tolist()

# -----------------------------
# 📁 Step 2: Load category examples
# -----------------------------
print("📁 Loading category examples...")
df_examples = pd.read_csv("category_examples.csv")
examples = df_examples["example"].fillna("").astype(str).tolist()
categories = df_examples["category"].tolist()

# -----------------------------
# 🧠 Step 3: Load embedding model
# -----------------------------
print("🧠 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
example_embeddings = model.encode(examples, normalize_embeddings=True)

# -----------------------------
# ⚡ Step 4: Build FAISS index
# -----------------------------
print("⚡ Building vector index...")
dimension = example_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(example_embeddings)

# -----------------------------
# 🧭 Step 5: Embed descriptions and classify
# -----------------------------
print("🧭 Classifying expense descriptions...")
desc_embeddings = model.encode(descriptions, normalize_embeddings=True)
scores, matched_indices = index.search(desc_embeddings, k=1)

# Add confidence flag here
LOW_CONFIDENCE_THRESHOLD = 0.45
df["low_confidence_flag"] = scores[:, 0] < LOW_CONFIDENCE_THRESHOLD

matched_categories = [categories[i[0]] for i in matched_indices]
matched_examples = [examples[i[0]] for i in matched_indices]

# -----------------------------
# 📝 Step 6: Add results to DataFrame
# -----------------------------
df["category"] = matched_categories
df["matched_example"] = matched_examples
df["similarity_score"] = scores[:, 0]

# -----------------------------
# 💾 Step 7: Save results
# -----------------------------
output_path = "data/expenses_categorized.csv"
df.to_csv(output_path, index=False)
print(f"✅ Categorized expenses saved to: {output_path}")

# -----------------------------
# 📊 Step 8: Preview top categories
# -----------------------------
print("\n📊 Top categories by count:")
print(df["category"].value_counts())
