
import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------
# STEP 1: Setup ChromaDB
# ----------------------
client = chromadb.Client()
collection_name = "expense_categories"
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(collection_name)
collection = client.create_collection(collection_name)

# ----------------------
# STEP 2: Define categories & examples
# ----------------------
category_examples = {
    "Groceries": ["walmart", "safeway", "trader joe", "grocery", "aldi"],
    "Eating Out": ["restaurant", "dinner", "lunch", "cafe", "burger king", "starbucks", "takeout"],
    "Alcohol": ["wine", "beer", "vodka", "liquor"],
    "Rent": ["monthly rent", "apartment", "lease"],
    "Utilities": ["electricity", "water bill", "utilities", "power", "gas"],
    "Travel": ["uber", "flight", "taxi", "bus", "train", "ola"],
    "Entertainment": ["movie", "netflix", "concert", "music", "amusement"],
    "Mobile Data": ["jio", "mint", "verizon", "mobile recharge"],
    "Clothes + Accessories": ["jeans", "zara", "jewelry", "tshirt", "dress"],
}

# ----------------------
# STEP 3: Embed examples into ChromaDB
# ----------------------
print("📥 Embedding category examples into ChromaDB...")
model = SentenceTransformer("all-MiniLM-L6-v2")
example_texts = []
example_ids = []
example_metas = []

for cat, examples in category_examples.items():
    for idx, ex in enumerate(examples):
        example_ids.append(f"{cat}_{idx}")
        example_texts.append(ex)
        example_metas.append({"category": cat})

collection.add(
    documents=example_texts,
    metadatas=example_metas,
    ids=example_ids,
    embeddings=model.encode(example_texts).tolist()
)

# ----------------------
# STEP 4: Load your expenses
# ----------------------
df = pd.read_csv("data/expenses.csv")
descriptions = df["description"].astype(str).fillna("").tolist()

# ----------------------
# STEP 5: Define keyword rules
# ----------------------
def keyword_rule(description):
    desc = description.lower()
    if any(k in desc for k in category_examples["Groceries"]): return "Groceries"
    if any(k in desc for k in category_examples["Eating Out"]): return "Eating Out"
    if any(k in desc for k in category_examples["Alcohol"]): return "Alcohol"
    if any(k in desc for k in category_examples["Rent"]): return "Rent"
    if any(k in desc for k in category_examples["Utilities"]): return "Utilities"
    if any(k in desc for k in category_examples["Travel"]): return "Travel"
    if any(k in desc for k in category_examples["Entertainment"]): return "Entertainment"
    if any(k in desc for k in category_examples["Mobile Data"]): return "Mobile Data"
    if any(k in desc for k in category_examples["Clothes + Accessories"]): return "Clothes + Accessories"
    return None

# ----------------------
# STEP 6: Categorize descriptions
# ----------------------
predicted = []
similarities = []
LOW_CONFIDENCE_THRESHOLD = 0.45

print("🔍 Classifying descriptions...")
for desc in tqdm(descriptions):
    rule_cat = keyword_rule(desc)
    if rule_cat:
        predicted.append(rule_cat)
        similarities.append(1.0)
        continue

    query_vec = model.encode(desc).tolist()
    result = collection.query(query_embeddings=[query_vec], n_results=1)
    best_match = result['metadatas'][0][0]['category']
    similarity_score = result['distances'][0][0]  # cosine distance

    # Convert cosine distance to similarity
    similarity = 1 - similarity_score
    if similarity < LOW_CONFIDENCE_THRESHOLD:
        predicted.append("Needs Review")
    else:
        predicted.append(best_match)
    similarities.append(round(similarity, 3))

df["predicted_category"] = predicted
df["similarity_score"] = similarities

# ----------------------
# STEP 7: Save results
# ----------------------
output_path = "data/expenses_with_categories.csv"
df.to_csv(output_path, index=False)
print(f"✅ Categorized data saved to {output_path}")
