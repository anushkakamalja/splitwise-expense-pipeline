import pandas as pd
import random
from faker import Faker
from collections import defaultdict

# Load original expenses data
df = pd.read_csv("data/expenses.csv")

# Step 1: Normalize and clean friend names
df["paid_by"] = df["paid_by"].astype(str)  # Avoid NaN issues
df["paid_by_clean"] = df["paid_by"].str.strip().str.lower()

# Check for null or empty cleaned names before mapping
null_or_empty = df["paid_by_clean"].isnull().sum() + (df["paid_by_clean"] == "").sum()
print(f"⚠️ Rows with null or empty 'paid_by_clean': {null_or_empty}")

# Step 2: Extract unique cleaned names (excluding empty strings)
unique_clean_names = df["paid_by_clean"].dropna()
unique_clean_names = unique_clean_names[unique_clean_names != ""].unique()
print(f"🔍 Found {len(unique_clean_names)} unique friend names.")

# Step 3: Generate unique fake names using Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

fake_names_pool = set()
while len(fake_names_pool) < len(unique_clean_names):
    fake_names_pool.add(fake.first_name())

fake_names_pool = list(fake_names_pool)
random.shuffle(fake_names_pool)

# Step 4: Create one-to-one mapping from cleaned name to fake name
name_map = dict(zip(unique_clean_names, fake_names_pool))

# Step 4.1: Check for cleaned names missing from mapping (should be none)
missing_in_map = set(df["paid_by_clean"].unique()) - set(name_map.keys())
missing_in_map = {name for name in missing_in_map if name not in [None, ""]}
print(f"⚠️ Cleaned names missing from mapping (should be empty): {missing_in_map}")

# Step 5: Apply fake names to original DataFrame using cleaned names
# For any missing or empty cleaned names, map to "Unknown" or leave as is
def map_name(x):
    if x in name_map:
        return name_map[x]
    else:
        return "Unknown"

df["paid_by_anon"] = df["paid_by_clean"].apply(map_name)

# Step 6: Validation of counts
print("\n🧪 Validating real vs fake name counts...")
errors_found = False
real_counts = df["paid_by_clean"].value_counts()
fake_counts = df["paid_by_anon"].value_counts()

for real_clean, fake_name in name_map.items():
    real_count = real_counts.get(real_clean, 0)
    fake_count = fake_counts.get(fake_name, 0)
    if real_count != fake_count:
        print(f"⚠️ Mismatch: '{real_clean}' → '{fake_name}': real={real_count}, fake={fake_count}")
        errors_found = True

if not errors_found:
    print("✅ All name mappings are consistent!")

# Step 7: Detect any duplicate fake name usage
reverse_map = defaultdict(list)
for real, fake in name_map.items():
    reverse_map[fake].append(real)

for fake_name, real_names in reverse_map.items():
    if len(real_names) > 1:
        print(f"❗ Duplicate fake name: {fake_name} is mapped to multiple real names: {real_names}")

# Step 8: Build anonymized dataframe with clean fake names
df_anonymized = df.copy()
df_anonymized["paid_by"] = df["paid_by_anon"]  # Override real with fake
df_anonymized = df_anonymized.drop(columns=["paid_by_clean", "paid_by_anon"])

# Step 9: Save anonymized version
df_anonymized.to_csv("data/expenses_anonymized.csv", index=False)
print("📁 Saved anonymized dataset to 'data/expenses_anonymized.csv'")

# Step 10: Save name mapping
name_mapping_df = pd.DataFrame({
    "Real Name (cleaned)": list(name_map.keys()),
    "Fake Name": list(name_map.values())
})
name_mapping_df.to_csv("data/name_mapping.csv", index=False)
print("📁 Saved mapping to 'data/name_mapping.csv'")
