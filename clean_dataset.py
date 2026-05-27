import pandas as pd

df = pd.read_csv("hinglish_toxic_dataset.csv")

print(f"Original dataset: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"\nLabel distribution (raw):")
print(df["label"].value_counts(dropna=False))

# 1. Keep only valid labels
valid_labels = ["offensive", "not_offensive"]
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(valid_labels)]

print(f"\nAfter removing invalid labels: {len(df)} rows")
print(df["label"].value_counts())

# 2. Clean text column
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 5]       # drop tiny/empty texts
df = df[df["text"].str.len() <= 1000]     # drop absurdly long texts
df = df[df["text"] != "nan"]

# 3. Drop duplicates
before_dedup = len(df)
df = df.drop_duplicates(subset=["text"], keep="first")
print(f"Dropped {before_dedup - len(df)} duplicate texts")

# 4. Final stats
print(f"\n--- FINAL CLEAN DATASET ---")
print(f"Total rows: {len(df)}")
print(f"Label distribution:")
print(df["label"].value_counts())
print(f"Offensive %: {(df['label'] == 'offensive').mean() * 100:.1f}%")
print(f"\nSample texts:")
for label in valid_labels:
    print(f"\n[{label}]:")
    samples = df[df["label"] == label].sample(3, random_state=42)
    for _, row in samples.iterrows():
        print(f"  → {row['text'][:100]}")

# 5. Save
df.to_csv("hinglish_toxic_clean.csv", index=False)
print(f"\nSaved to: hinglish_toxic_clean.csv")
