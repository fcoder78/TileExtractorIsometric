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

# Full sprite image size (must be exact for every final export)
TILE_WIDTH = 512
TILE_HEIGHT = 256

MIN_TILE_AREA = 12000
REPROCESS_ONLY = False  # True = only apply anchor overrides to existing exports

# Background / mask tuning
BORDER_THICKNESS = 16
BG_DISTANCE_THRESHOLD = 18
MORPH_KERNEL_SIZE = 3
MAX_COMPONENT_GAP_FILL = 3
COMPONENT_PADDING = 16

# Export tuning
# Anchor must stay inside the final 512x256 canvas
EXPORT_ANCHOR_X = TILE_WIDTH // 2
EXPORT_ANCHOR_Y = TILE_HEIGHT - 1  # bottom-center anchor

FIXED_CANVAS_WIDTH = TILE_WIDTH
FIXED_CANVAS_HEIGHT = TILE_HEIGHT

# Contact sheet
CONTACT_SHEET_COLS = 4
THUMB_W = 256
THUMB_H = 256


# ============================================================
# HELPERS
# ============================================================

def ensure_rgba(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def sample_border_pixels(img_bgr, border_thickness):
    h, w = img_bgr.shape[:2]
    t = min(border_thickness, max(1, h // 4), max(1, w // 4))

    top = img_bgr[:t, :, :].reshape(-1, 3)
    bottom = img_bgr[h - t:h, :, :].reshape(-1, 3)
    left = img_bgr[:, :t, :].reshape(-1, 3)
    right = img_bgr[:, w - t:w, :].reshape(-1, 3)

    return np.vstack([top, bottom, left, right])


def estimate_background_color(img_bgr):
    border_pixels = sample_border_pixels(img_bgr, BORDER_THICKNESS)
    return np.median(border_pixels, axis=0).astype(np.uint8)


def build_foreground_mask(img):
    """
    Build a foreground mask by comparing every pixel to the estimated
    background color from the image borders.
    """
    bgr = img[:, :, :3]
    bg_color = estimate_background_color(bgr)

    img_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    bg_patch = np.full((1, 1, 3), bg_color, dtype=np.uint8)
    bg_lab = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.int16)

    diff = img_lab.astype(np.int16) - bg_lab
    dist = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    mask = np.zeros(dist.shape, dtype=np.uint8)
    mask[dist >= BG_DISTANCE_THRESHOLD] = 255

    # Less aggressive cleanup so thin details are preserved
    k = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if MAX_COMPONENT_GAP_FILL > 0:
        kx = np.ones((1, MAX_COMPONENT_GAP_FILL), np.uint8)
        ky = np.ones((MAX_COMPONENT_GAP_FILL, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kx)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ky)

    return mask, bg_color


def find_components(mask):
    """
    Return padded bounding boxes and masks for connected components
    larger than MIN_TILE_AREA.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    img_h, img_w = mask.shape

    components = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < MIN_TILE_AREA:
            continue

        x0 = max(0, x - COMPONENT_PADDING)
        y0 = max(0, y - COMPONENT_PADDING)
        x1 = min(img_w, x + w + COMPONENT_PADDING)
        y1 = min(img_h, y + h + COMPONENT_PADDING)

        padded_w = x1 - x0
        padded_h = y1 - y0

        component_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
        region = labels[y0:y1, x0:x1]
        component_mask[region == label] = 255

        components.append({
            "bbox": [x0, y0, padded_w, padded_h],
            "area": area,
            "mask": component_mask
        })

    return sort_components_reading_order(components)


def sort_components_reading_order(components):
    if not components:
        return components

    avg_h = int(np.mean([c["bbox"][3] for c in components]))
    row_bucket = max(80, avg_h // 2)

    return sorted(
        components,
        key=lambda c: (c["bbox"][1] // row_bucket, c["bbox"][0])
    )


def estimate_anchor(mask):
    """
    Estimate anchor as the lowest occupied row's horizontal center.
    """
    h, w = mask.shape
    for y in range(h - 1, -1, -1):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) > 0:
            return int((xs[0] + xs[-1]) / 2), int(y)
    return w // 2, h - 1


def save_mask_debug(mask, output_path):
    Image.fromarray(mask).save(output_path)


def save_components_debug(img, components, output_path):
    dbg = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    dbg = Image.fromarray(dbg)
    draw = ImageDraw.Draw(dbg)

    for i, comp in enumerate(components):
        x, y, w, h = comp["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
        draw.text((x + 4, y + 4), f"{i:03d}", fill=(255, 255, 0))

    dbg.save(output_path)


def create_rgba_crop(img, bbox, local_mask):
    x, y, w, h = bbox
    crop = img[y:y + h, x:x + w].copy()

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(rgba)
    pil_img.putalpha(Image.fromarray(local_mask))
    return pil_img


def check_tile_fits_canvas(local_mask, anchor_x, anchor_y):
    """
    Check whether the extracted object fits fully inside the final fixed canvas
    when placed at the export anchor.
    """
    h, w = local_mask.shape

    left = anchor_x
    right = w - anchor_x - 1
    above = anchor_y
    below = h - anchor_y - 1

    fits_horizontally = (
        left <= EXPORT_ANCHOR_X and
        right <= (FIXED_CANVAS_WIDTH - EXPORT_ANCHOR_X - 1)
    )
    fits_vertically = (
        above <= EXPORT_ANCHOR_Y and
        below <= (FIXED_CANVAS_HEIGHT - EXPORT_ANCHOR_Y - 1)
    )

    return fits_horizontally and fits_vertically, {
        "left": int(left),
        "right": int(right),
        "above": int(above),
        "below": int(below),
    }


def normalize_tile_to_canvas(tile_img, local_mask, anchor_x, anchor_y):
    """
    Normalize every object to the exact same final canvas size.
    Every exported PNG will be exactly 512x256.
    """
    canvas_w = FIXED_CANVAS_WIDTH
    canvas_h = FIXED_CANVAS_HEIGHT

    anchor_in_canvas_x = EXPORT_ANCHOR_X
    anchor_in_canvas_y = EXPORT_ANCHOR_Y

    paste_x = anchor_in_canvas_x - anchor_x
    paste_y = anchor_in_canvas_y - anchor_y

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(tile_img, (paste_x, paste_y), tile_img)

    return canvas, paste_x, paste_y, anchor_in_canvas_x, anchor_in_canvas_y


def create_contact_sheet(metadata, output_dir, debug_dir):
    if not metadata:
        return

    cols = CONTACT_SHEET_COLS
    rows = int(np.ceil(len(metadata) / cols))

    sheet = Image.new(
        "RGBA",
        (cols * THUMB_W, rows * THUMB_H),
        (30, 30, 30, 255)
    )
    draw = ImageDraw.Draw(sheet)

    for i, item in enumerate(metadata):
        path = os.path.join(output_dir, item["file"])
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert("RGBA")
        thumb = img.resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)

        x = (i % cols) * THUMB_W
        y = (i // cols) * THUMB_H

        sheet.paste(thumb, (x, y), thumb)
        draw.text((x + 6, y + 6), item["id"], fill=(255, 255, 0))

        anchor_canvas_x = item["export_anchor"][0]
        anchor_canvas_y = item["export_anchor"][1]

        ax = int((anchor_canvas_x / max(img.width, 1)) * THUMB_W)
        ay = int((anchor_canvas_y / max(img.height, 1)) * THUMB_H)

        draw.line((x + ax - 6, y + ay, x + ax + 6, y + ay), fill="red", width=2)
        draw.line((x + ax, y + ay - 6, x + ax, y + ay + 6), fill="red", width=2)

    sheet.save(os.path.join(debug_dir, "contact_sheet.png"))
    print("[INFO] Contact sheet generated")


# ============================================================
# CORE PROCESSING
# ============================================================

def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    output_dir = os.path.join(OUTPUT_ROOT, image_name)
    debug_dir = os.path.join(output_dir, "_debug")

    meta_file = os.path.join(output_dir, "metadata.json")
    anchor_override_file = os.path.join(output_dir, "anchors_override.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"\n[PROCESSING] {image_name}")

    if REPROCESS_ONLY and os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        reprocess(metadata, output_dir, anchor_override_file, meta_file)
        create_contact_sheet(metadata, output_dir, debug_dir)
        return

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = ensure_rgba(img)

    if img is None:
        print(f"[ERROR] Failed to load {image_path}")
        return

    mask, bg_color = build_foreground_mask(img)
    components = find_components(mask)

    print(f"[INFO] Estimated background color: {bg_color.tolist()}")
    print(f"[INFO] Detected components: {len(components)}")

    save_mask_debug(mask, os.path.join(debug_dir, "mask.png"))
    save_components_debug(img, components, os.path.join(debug_dir, "components.png"))

    metadata = []

    for i, comp in enumerate(components):
        x, y, w, h = comp["bbox"]
        local_mask = comp["mask"]

        anchor_x, anchor_y = estimate_anchor(local_mask)
        tile_img = create_rgba_crop(img, comp["bbox"], local_mask)

        fits, extents = check_tile_fits_canvas(local_mask, anchor_x, anchor_y)
        if not fits:
            print(
                f"[WARNING] object_{i:03d} may be clipped in {FIXED_CANVAS_WIDTH}x{FIXED_CANVAS_HEIGHT}. "
                f"Extents={extents}, export_anchor=({EXPORT_ANCHOR_X}, {EXPORT_ANCHOR_Y})"
            )

        canvas, paste_x, paste_y, export_anchor_x, export_anchor_y = normalize_tile_to_canvas(
            tile_img, local_mask, anchor_x, anchor_y
        )

        name = f"object_{i:03d}.png"
        save_path = os.path.join(output_dir, name)
        canvas.save(save_path)

        metadata.append({
            "id": f"object_{i:03d}",
            "file": name,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(comp["area"]),
            "source_anchor": [int(anchor_x), int(anchor_y)],
            "paste": [int(paste_x), int(paste_y)],
            "export_anchor": [int(export_anchor_x), int(export_anchor_y)],
            "logical_tile_size": [int(TILE_WIDTH), int(TILE_HEIGHT)],
            "export_size": [int(FIXED_CANVAS_WIDTH), int(FIXED_CANVAS_HEIGHT)],
            "fits_fixed_canvas": bool(fits),
            "extents_from_anchor": extents
        })

        print(
            f"[OK] {name} bbox=({x}, {y}, {w}, {h}) "
            f"area={comp['area']} export_size=({FIXED_CANVAS_WIDTH}, {FIXED_CANVAS_HEIGHT}) "
            f"export_anchor=({export_anchor_x}, {export_anchor_y})"
        )

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not os.path.exists(anchor_override_file):
        with open(anchor_override_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    create_contact_sheet(metadata, output_dir, debug_dir)
    print(f"[DONE] {image_name}")


# ============================================================
# REPROCESS (WITH OVERRIDES)
# ============================================================

def reprocess(metadata, output_dir, anchor_override_file, meta_file):
    if os.path.exists(anchor_override_file):
        with open(anchor_override_file, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    else:
        overrides = {}

    for item in metadata:
        path = os.path.join(output_dir, item["file"])
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert("RGBA")

        old_anchor_x, old_anchor_y = item.get(
            "export_anchor",
            [EXPORT_ANCHOR_X, EXPORT_ANCHOR_Y]
        )

        export_anchor_x = old_anchor_x
        export_anchor_y = old_anchor_y

        if item["id"] in overrides:
            export_anchor_x = overrides[item["id"]].get("anchor_x", old_anchor_x)
            export_anchor_y = overrides[item["id"]].get("anchor_y", old_anchor_y)

        shift_x = export_anchor_x - old_anchor_x
        shift_y = export_anchor_y - old_anchor_y

        canvas = Image.new(
            "RGBA",
            (FIXED_CANVAS_WIDTH, FIXED_CANVAS_HEIGHT),
            (0, 0, 0, 0)
        )
        canvas.paste(img, (shift_x, shift_y), img)
        canvas.save(path)

        item["export_anchor"] = [int(export_anchor_x), int(export_anchor_y)]
        item["export_size"] = [int(FIXED_CANVAS_WIDTH), int(FIXED_CANVAS_HEIGHT)]

        print(f"[UPDATED] {item['file']}")

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
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