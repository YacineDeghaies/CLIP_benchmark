from transformers import pipeline

import json
import os
import time

#create a Pipeline & use a batch_size to improve speed
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

#initialize a list to collect all translated captions for later use
translated = []
batch_size=64

#construct the path to the original de captions file
caption_x_file = "/vol/fob-vol3/mi20/deghaisa/code/CLIP_benchmark/clip_benchmark/root/crossmodal3600_captions-de.json"

#open the file
with open(caption_x_file) as f:
     #load the file
     data = json.load(f)
     #extract original captions - no need to iterate through a loop for this
     captions = data["annotations"]
     for i in range(0, len(captions), batch_size):
          batch = captions[i:i+batch_size]
          print(f"batch number {i}:", batch)
          print(f"batch size {len(batch)}:")
          start_t = time.time()
          results = translator(batch)
          end_t = time.time()
          print(f"Execution time of each batch {i}:{end_t - start_t} seconds")
          translated.extend([r["translation_text"] for r in results])
     
#dump all from x to english translated captions into a .json file
with open("captions_x_to_eng.json", "w", encoding="utf-8") as cf:
     json.dump(
     {
          "image_paths":data["image_paths"],
          "annotations":translated,
          "indicies":data["indicies"]
     },
     cf,
     ensure_ascii=False,
     )
     