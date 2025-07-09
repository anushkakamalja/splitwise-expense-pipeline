import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# 🔹 Step 1: Load test set and example set
print("📥 Loading test and examples...")
df_test = pd.read_csv("data/test_labels.csv")  # Must have 'description' and 'true_category'
df_examples = pd.read_csv("data/category_examples.csv")  # 'category', 'example'

# 🔹 Step 2: Load embedding model
print("🧠 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Step 3: Embed examples
category_labels = df_examples["category"].unique().tolist()
category_to_examples = {cat: df_examples[df_examples["category"] == cat]["example"].tolist() for cat in category_labels}

category_embeddings = {}
for category, examples in category_to_examples.items():
    embeddings = model.encode(examples, convert_to_tensor=True)
    avg_embedding = embeddings.mean(dim=0)
    category_embeddings[category] = avg_embedding

# 🔹 Step 4: Embed and predict
print("🔍 Classifying descriptions...")
desc_embeddings = model.encode(df_test["description"].tolist(), convert_to_tensor=True)

predicted = []
similarities = []

for i, desc_emb in tqdm(enumerate(desc_embeddings), total=len(desc_embeddings)):
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

# 🔹 Step 5: Add confidence bucket
def confidence_bucket(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.45:
        return "Medium"
    else:
        return "Low"

df_test["confidence_bucket"] = df_test["similarity_score"].apply(confidence_bucket)

# 🔹 Step 6: Accuracy
df_test["correct"] = df_test["true_category"].str.lower().str.strip() == df_test["predicted_category"].str.lower().str.strip()

accuracy = df_test["correct"].mean()
low_conf_count = (df_test["confidence_bucket"] == "Low").sum()
incorrect_count = (~df_test["correct"]).sum()

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print(f"⚠️  Low Confidence Predictions (< 0.45): {low_conf_count}")
print(f"❌ Incorrect Predictions: {incorrect_count}")

# 🔹 Step 7: Per-category breakdown
print("\n📊 Per-Category Accuracy:")
print(df_test.groupby("true_category")["correct"].agg(["count", "sum", "mean"]).rename(
    columns={"sum": "correct", "mean": "accuracy"}).round(2)
)

# 🔹 Step 8: Save incorrect / low-confidence examples
errors_df = df_test[(~df_test["correct"]) | (df_test["confidence_bucket"] == "Low")]
errors_df.to_csv("data/eval_errors.csv", index=False)
print("\n📁 Saved mismatches and low-confidence rows to: data/eval_errors.csv")
