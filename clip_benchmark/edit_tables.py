import pandas as pd

# --- Load the original CSV ---
file_path = "/vol/fob-vol3/mi20/deghaisa/code/CLIP_benchmark/clip_benchmark/fr_to_eng_benchmark/fr_to_en_benchmark.csv"
df = pd.read_csv(file_path)

# --- 1.  Add the average Recall column ---
df["avg"] = (
    df["image_retrieval_recall@1"]
  + df["image_retrieval_recall@5"]
  + df["image_retrieval_recall@10"]
) / 3.0

# --- 2.  Round numeric columns to 4 decimal places ---
cols_to_round = ["avg",
                 "image_retrieval_recall@1",
                 "image_retrieval_recall@5",
                 "image_retrieval_recall@10"]
df[cols_to_round] = df[cols_to_round].round(4)

# --- 3. Re-order columns to match the desired table ---
df = df[["model",
         "pretrained",
         "avg",
         "image_retrieval_recall@1",
         "image_retrieval_recall@5",
         "image_retrieval_recall@10"]]

# --- 4.  Save back to CSV (overwrite or change name) ---
df.to_csv(file_path, index=False)
print(f"Saved updated file to: {file_path}")