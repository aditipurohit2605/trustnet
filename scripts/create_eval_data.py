"""
Generate gold OCR evaluation data and reference images for TrustNet Stage 5.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

img_dir = Path("data/eval/images")
img_dir.mkdir(parents=True, exist_ok=True)

# Select standard Windows fonts
try:
    font_en = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
except Exception:
    font_en = ImageFont.load_default()

try:
    font_hi = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 26)
except Exception:
    font_hi = font_en

samples = [
    {"id": "ocr_en_1", "text": "The dam reservoir levels are currently normal and safe.", "language": "en", "font": font_en},
    {"id": "ocr_en_2", "text": "Official notice: please do not spread unverified rumors.", "language": "en", "font": font_en},
    {"id": "ocr_en_3", "text": "The emergency response team has been dispatched.", "language": "en", "font": font_en},
    {"id": "ocr_en_4", "text": "Weather department predicts light rain tomorrow morning.", "language": "en", "font": font_en},
    {"id": "ocr_hi_1", "text": "बाढ़ की स्थिति पूरी तरह से नियंत्रण में है", "language": "hi", "font": font_hi},
    {"id": "ocr_hi_2", "text": "आधिकारिक सूचना: किसी भी अफवाह पर ध्यान न दें", "language": "hi", "font": font_hi},
    {"id": "ocr_hi_3", "text": "सुरक्षा हेल्पलाइन नंबर चालू है", "language": "hi", "font": font_hi},
    {"id": "ocr_hi_4", "text": "मौसम विभाग ने हल्की बारिश का अनुमान लगाया है", "language": "hi", "font": font_hi},
]

records = []
for s in samples:
    img_name = f"{s['id']}.png"
    img_path = img_dir / img_name
    img = Image.new("RGB", (750, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((25, 40), s["text"], font=s["font"], fill=(0, 0, 0))
    img.save(img_path)
    records.append({
        "id": s["id"],
        "text": s["text"],
        "language": s["language"],
        "image_path": str(img_path).replace("\\", "/"),
    })

gold_path = Path("data/eval/ocr_gold.jsonl")
with open(gold_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Generated {len(records)} gold OCR samples and saved to {gold_path}")
