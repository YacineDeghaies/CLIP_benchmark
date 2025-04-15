from transformers import pipeline
import json
import os
import time

#create a Pipeline & use a batch_size to improve speed
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en", batch_size=64)

#test the Pipeline
print(type(translator("Ich liebe Clip-Netzwerke!")))
#see a sample looks like
     #extract the translated text only
print(translator("Ich liebe Clip-Netzwerke!")[0]["translation_text"])

#initialize a list to collect all translated captions for later use
all_trans_captions = []

#construct path to the captions file
caption_x_file = "/vol/fob-vol3/mi20/deghaisa/code/CLIP_benchmark/clip_benchmark/root/crossmodal3600_captions-de.json"

     #open it
with open(caption_x_file) as f:
     #load its data using json
     data = json.load(f)
     #iterate through each caption
     for ann in data["annotations"]:
          #translate each caption
          current_trans_caption = translator(ann)[0]["translation_text"]
          #append each translated caption
          all_trans_captions.append(current_trans_caption)
          
#dump all from x to english translated captions into a .json file
with open("captions_x_to_eng.json", "w") as cf:
     json.dump(
     {
          "annotations":all_trans_captions
     },
     cf,
     ensure_ascii=False,
     )
               
