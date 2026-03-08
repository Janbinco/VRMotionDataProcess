"""
Split multiview images (2x2 grid) into 4 individual views.
For each PNG directly inside unityresults/<variant>/:
  -> creates unityresults/<variant>/mv_<stem>/tl.png, tr.png, bl.png, br.png

Skips: singleview folders, existing mv_* folders, temp_processed folders.
"""

from pathlib import Path
from PIL import Image

BASE_DIR = Path(r"C:\Users\janbi\OneDrive - George Mason University - O365 Production\Project2\unity_gemini_mesh")

SKIP_VARIANTS = {"singleview", "temp_processed"}

processed = 0
skipped = 0

for unityresults_dir in BASE_DIR.rglob("unityresults"):
    if not unityresults_dir.is_dir():
        continue

    for variant_dir in sorted(unityresults_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        if variant_dir.name in SKIP_VARIANTS:
            print(f"  [SKIP folder] {variant_dir.relative_to(BASE_DIR)}")
            skipped += 1
            continue
        if variant_dir.name.startswith("mv_"):
            continue

        png_files = sorted(variant_dir.glob("*.png"))
        if not png_files:
            continue

        for png_file in png_files:
            stem = png_file.stem
            out_dir = variant_dir / f"mv_{stem}"

            if out_dir.exists():
                print(f"  [SKIP exists] {out_dir.relative_to(BASE_DIR)}")
                skipped += 1
                continue

            out_dir.mkdir()
            img = Image.open(png_file)
            w, h = img.size
            hw, hh = w // 2, h // 2

            crops = {
                "tl": img.crop((0,  0,  hw, hh)),
                "tr": img.crop((hw, 0,  w,  hh)),
                "bl": img.crop((0,  hh, hw, h)),
                "br": img.crop((hw, hh, w,  h)),
            }
            for name, crop in crops.items():
                crop.save(out_dir / f"{name}.png")

            print(f"  [OK] {png_file.relative_to(BASE_DIR)}  ->  mv_{stem}/{{tl,tr,bl,br}}.png")
            processed += 1

print(f"\nDone. Processed: {processed}  Skipped: {skipped}")
