"""
Populates data/image_bank with initial reference images and metadata.jsonl for TrustNet Stage 5.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw

bank_dir = Path("data/image_bank")
bank_dir.mkdir(parents=True, exist_ok=True)

references = [
    {
        "bank_id": "bank_img_001",
        "title": "Dam Reservoir Safe Water Levels",
        "known_context": "Official state reservoir camera feed verifying normal water levels at Bhakra Nangal Dam.",
        "source": "State Water Resources Information System (Official)",
        "color": (40, 120, 200),
    },
    {
        "bank_id": "bank_img_002",
        "title": "Railway Bridge Construction",
        "known_context": "Construction progress photo of the Chenab rail arch bridge in Jammu and Kashmir.",
        "source": "Ministry of Railways Press Information Bureau",
        "color": (180, 80, 40),
    },
    {
        "bank_id": "bank_img_003",
        "title": "Solar Power Park Grid",
        "known_context": "Aerial view of Bhadla Solar Park arrays in Rajasthan.",
        "source": "Renewable Energy Authority Report 2024",
        "color": (200, 160, 30),
    },
    {
        "bank_id": "bank_img_004",
        "title": "Disaster Relief Food Distribution",
        "known_context": "National Disaster Response Force (NDRF) team distributing relief packets during 2023 floods.",
        "source": "NDRF Official Media Handle",
        "color": (50, 150, 80),
    },
    {
        "bank_id": "bank_img_005",
        "title": "Hospital Emergency Oxygen Plant",
        "known_context": "Pressure Swing Adsorption (PSA) medical oxygen generation plant inaugurated in Delhi.",
        "source": "Ministry of Health and Family Welfare",
        "color": (130, 60, 160),
    },
]

metadata_lines = []
for ref in references:
    bid = ref["bank_id"]
    img_path = bank_dir / f"{bid}.png"

    # Create visually distinct synthetic scene
    img = Image.new("RGB", (320, 240), color=ref["color"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 300, 220], outline=(255, 255, 255), width=3)
    draw.text((30, 40), ref["title"], fill=(255, 255, 255))
    draw.text((30, 80), f"ID: {bid}", fill=(240, 240, 240))
    img.save(img_path)

    metadata_lines.append({
        "bank_id": bid,
        "path": str(img_path).replace("\\", "/"),
        "known_context": ref["known_context"],
        "source": ref["source"],
    })

meta_file = bank_dir / "metadata.jsonl"
with open(meta_file, "w", encoding="utf-8") as f:
    for m in metadata_lines:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"Created {len(references)} bank images and metadata at {bank_dir}")
