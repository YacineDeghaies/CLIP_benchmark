import os, json
import deepl
from pathlib import Path

auth = os.getenv("DEEPL_AUTH_KEY")

translator = deepl.Translator(auth)
SOURCE_LANG, TARGET_LANG = "DE", "EN-US"

batch_size = 50                          
translated = []

captions_file_path = "/vol/fob-vol3/mi20/deghaisa/code/CLIP_benchmark/clip_benchmark/root/crossmodal3600_captions-de.json"
with open(captions_file_path) as f:
    data = json.load(f)
    captions = data["annotations"]

for i in range(0, len(captions), batch_size):
    batch = captions[i : i + batch_size]
    results = translator.translate_text(
        batch,
        # source_lang=None, 
        target_lang=TARGET_LANG,
    )
    translated.extend(r.text for r in results)

# ensure output directory exists
out_dir = Path("./api_translated_annotation_files")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "captions_de_to_en.json", "w", encoding="utf-8") as cf:
    json.dump(
        {
            "image_paths": data["image_paths"],
            "annotations": translated,
            "indicies": data["indicies"],
        },
        cf,
        ensure_ascii=False,
    )