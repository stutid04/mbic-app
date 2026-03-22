import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# 1) Load Excel
df = pd.read_excel("labeled_dataset.xlsx")

# 2) Keep only needed columns
df = df[["sentence", "Label_bias", "outlet"]].dropna(subset=["sentence", "Label_bias"])

# 3) Simplify labels
label_map = {"Non-biased": 0, "Biased": 1}
df = df[df["Label_bias"].isin(label_map)]
df["label"] = df["Label_bias"].map(label_map)

# 4) Basic cleanup
df["sentence"] = df["sentence"].str.replace(r"\s+", " ", regex=True).str.strip()

# 5) Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# 6) Create Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df[["sentence", "label"]].reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df[["sentence", "label"]].reset_index(drop=True))

print(train_ds, test_ds)
train_ds[0]