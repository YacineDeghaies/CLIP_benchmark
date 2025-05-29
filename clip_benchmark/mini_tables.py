import pandas as pd

df = pd.read_csv("/vol/fob-vol3/mi20/deghaisa/code/CLIP_benchmark/clip_benchmark/de_to_eng_benchmark/de_to_en_text_to_image_only.csv")
df_top3    = df.sort_values("avg", ascending=False).head(3)
df_bottom3 = df.sort_values("avg", ascending=False).tail(3)
focus      = pd.concat([df_top3, df_bottom3])
focus.round(4).to_csv("mini_table_de_en_top_bottom.csv", index=False)