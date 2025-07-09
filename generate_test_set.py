import pandas as pd

# Load the original (non-anonymized) expenses dataset
df = pd.read_csv("data/expenses.csv")

# Drop rows with missing descriptions just in case
df = df.dropna(subset=["description"])

# Sample 100 random expense descriptions
test_sample = df[["description"]].drop_duplicates().sample(n=100, random_state=42).reset_index(drop=True)

# Add empty column for you to fill in manually
test_sample["true_category"] = ""

# Save to test_labels.csv for manual labeling
test_sample.to_csv("test_labels.csv", index=False)
print("✅ Saved test set with 100 random descriptions to 'test_labels.csv'")
