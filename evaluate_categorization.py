import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 🔹 Step 1: Load test set and category examples
print("📥 Loading data...")
df_test = pd.read_csv("data/test_labels.csv")
df_examples = pd.read_csv("data/category_examples.csv")

# 🔹 Step 2: Load sentence embedding model
print("🧠 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Step 3: Compute embeddings
category_labels = df_examples["category"].unique().tolist()
category_to_examples = {cat: df_examples[df_examples["category"] == cat]["example"].tolist() for cat in category_labels}

# Compute average embedding for each category
category_embeddings = {}
for category, examples in category_to_examples.items():
    embeddings = model.encode(examples, convert_to_tensor=True)
    avg_embedding = embeddings.mean(dim=0)
    category_embeddings[category] = avg_embedding

# 🔹 Step 4: Embed test descriptions and predict
desc_embeddings = model.encode(df_test["description"].tolist(), convert_to_tensor=True)

predicted = []
similarities = []

print("🔍 Classifying descriptions...")
for i, desc_emb in enumerate(desc_embeddings):
    max_sim = -1
    best_category = None
    for category, emb in category_embeddings.items():
        sim = util.pytorch_cos_sim(desc_emb, emb).item()
        if sim > max_sim:
            max_sim = sim
            best_category = category
    predicted.append(best_category)
    similarities.append(max_sim)

df_test["predicted_category"] = predicted
df_test["similarity_score"] = similarities

# 🔹 Step 5: Confidence scoring
def confidence_bucket(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.45:
        return "Medium"
    else:
        return "Low"

df_test["confidence_bucket"] = df_test["similarity_score"].apply(confidence_bucket)

# 🔹 Step 6: Evaluation
df_test["correct"] = df_test["true_category"].str.lower().str.strip() == df_test["predicted_category"].str.lower().str.strip()

accuracy = df_test["correct"].mean()
low_conf_count = (df_test["confidence_bucket"] == "Low").sum()
incorrect_count = (~df_test["correct"]).sum()

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print(f"⚠️  Low Confidence Predictions (< 0.45): {low_conf_count}")
print(f"❌ Incorrect Predictions: {incorrect_count}\n")

# 🔹 Step 7: Category-level metrics
print("📊 Per-Category Accuracy:")
print(df_test.groupby("true_category")["correct"].agg(["count", "sum", "mean"]).rename(columns={"sum": "correct", "mean": "accuracy"}).round(2))

# 🔹 Step 8: Save errors and low-confidence cases
errors_df = df_test[(~df_test["correct"]) | (df_test["confidence_bucket"] == "Low")]
errors_df.to_csv("data/eval_errors.csv", index=False)
print("\n📁 Mismatches/Low confidence saved to: data/eval_errors.csv")
