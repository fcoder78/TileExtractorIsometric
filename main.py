import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "input"
OUTPUT_ROOT = "output"

TILE_WIDTH = 512
TILE_HEIGHT = 256

MIN_TILE_AREA = 12000
REPROCESS_ONLY = False  # 🔥 set True to only reprocess using overrides

# ============================================================
# HELPERS
# ============================================================

def get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_tiles(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > MIN_TILE_AREA]

def estimate_anchor(mask):
    h, w = mask.shape
    for y in range(h - 1, 0, -1):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) > 0:
            return int((xs[0] + xs[-1]) / 2), y
    return w // 2, h - 1

# ============================================================
# CORE PROCESSING
# ============================================================

def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, image_name)
    DEBUG_DIR = os.path.join(OUTPUT_DIR, "_debug")

    META_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
    ANCHOR_OVERRIDE_FILE = os.path.join(OUTPUT_DIR, "anchors_override.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    print(f"\n[PROCESSING] {image_name}")

    if REPROCESS_ONLY and os.path.exists(META_FILE):
        with open(META_FILE) as f:
            metadata = json.load(f)
        reprocess(metadata, OUTPUT_DIR, ANCHOR_OVERRIDE_FILE)
        create_contact_sheet(metadata, OUTPUT_DIR, DEBUG_DIR)
        return

    # =========================
    # LOAD IMAGE
    # =========================
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] Failed to load {image_path}")
        return

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # =========================
    # DETECT TILES
    # =========================
    mask = get_mask(img)
    contours = find_tiles(mask)

    metadata = []

    # =========================
    # PROCESS EACH TILE
    # =========================
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        crop = img[y:y+h, x:x+w]
        local_mask = np.zeros((h, w), dtype=np.uint8)

        shifted = cnt.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y

        cv2.drawContours(local_mask, [shifted], -1, 255, -1)

        anchor_x, anchor_y = estimate_anchor(local_mask)

        canvas_h = max(420, h + 40)
        canvas = Image.new("RGBA", (TILE_WIDTH, canvas_h), (0, 0, 0, 0))

        tile_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA))
        tile_img.putalpha(Image.fromarray(local_mask))

        paste_x = (TILE_WIDTH // 2) - anchor_x
        paste_y = TILE_HEIGHT - anchor_y

        canvas.paste(tile_img, (paste_x, paste_y), tile_img)

        name = f"object_{i:03d}.png"
        canvas.save(os.path.join(OUTPUT_DIR, name))

        metadata.append({
            "id": f"object_{i:03d}",
            "file": name,
            "bbox": [int(x), int(y), int(w), int(h)],
            "anchor": [int(anchor_x), int(anchor_y)],
            "paste": [int(paste_x), int(paste_y)]
        })

        print(f"[OK] {name}")

    # =========================
    # SAVE METADATA
    # =========================
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    if not os.path.exists(ANCHOR_OVERRIDE_FILE):
        with open(ANCHOR_OVERRIDE_FILE, "w") as f:
            json.dump({}, f, indent=2)

    create_contact_sheet(metadata, OUTPUT_DIR, DEBUG_DIR)

    print(f"[DONE] {image_name}")


# ============================================================
# REPROCESS (WITH OVERRIDES)
# ============================================================

def reprocess(metadata, OUTPUT_DIR, ANCHOR_OVERRIDE_FILE):
    with open(ANCHOR_OVERRIDE_FILE) as f:
        overrides = json.load(f)

    for item in metadata:
        path = os.path.join(OUTPUT_DIR, item["file"])
        img = Image.open(path).convert("RGBA")

        anchor_x, anchor_y = item["anchor"]

        if item["id"] in overrides:
            anchor_x = overrides[item["id"]]["anchor_x"]
            anchor_y = overrides[item["id"]]["anchor_y"]

        canvas = Image.new("RGBA", (TILE_WIDTH, img.height), (0, 0, 0, 0))

        paste_x = (TILE_WIDTH // 2) - anchor_x
        paste_y = TILE_HEIGHT - anchor_y

        canvas.paste(img, (paste_x, paste_y), img)
        canvas.save(path)

        print(f"[UPDATED] {item['file']}")


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(metadata, OUTPUT_DIR, DEBUG_DIR):
    cols = 4
    rows = int(np.ceil(len(metadata) / cols))

    thumb_w = 256
    thumb_h = 256

    sheet = Image.new("RGBA", (cols * thumb_w, rows * thumb_h), (30, 30, 30, 255))
    draw = ImageDraw.Draw(sheet)

    for i, item in enumerate(metadata):
        img = Image.open(os.path.join(OUTPUT_DIR, item["file"])).resize((thumb_w, thumb_h))

        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h

        sheet.paste(img, (x, y), img)

        draw.text((x + 5, y + 5), item["id"], fill=(255, 255, 0))

        ax = thumb_w // 2
        ay = int((TILE_HEIGHT / img.height) * thumb_h)

        draw.line((x + ax - 5, y + ay, x + ax + 5, y + ay), fill="red", width=2)
        draw.line((x + ax, y + ay - 5, x + ax, y + ay + 5), fill="red", width=2)

    sheet.save(os.path.join(DEBUG_DIR, "contact_sheet.png"))
    print("[INFO] Contact sheet generated")


# ============================================================
# MAIN
# ============================================================

def main():
    # Ensure input folder exists
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"[INFO] Created '{INPUT_DIR}' folder. Put your images there and run again.")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not files:
        print("[WARNING] No images found in 'input/' folder.")
        print("Add images and run again.")
        return

    for file in files:
        process_image(os.path.join(INPUT_DIR, file))

    print("\nALL DONE")


if __name__ == "__main__":
    main()