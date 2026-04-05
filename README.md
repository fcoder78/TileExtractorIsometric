# Isometric Tile Extractor

This project provides a Python script that automatically extracts irregularly placed isometric tile objects from a source image such as `tiles.png`.

## Goal

The goal of this script is to turn a single mixed tileset image into individual PNG files that can be used in a diamond-based isometric renderer.

It is designed for tilesets where:

- tiles are **not** arranged in a regular grid
- tiles include **vertical depth** and painted side walls
- some objects are taller than others, such as trees or rocks
- each object must be exported as its own transparent image
- all exported images must align consistently in an isometric map

## What the script does

The extractor follows this workflow:

1. Load the input image
2. Detect separate visual objects using image processing
3. Crop each detected object
4. Estimate an anchor point for each object
5. Place each object onto a normalized transparent canvas
6. Export each result as a separate PNG with dynamic names such as:
   - `object_000.png`
   - `object_001.png`
   - `object_002.png`
7. Save metadata about the extraction
8. Generate a debug contact sheet for visual review

## Why this is useful

In an isometric game, each tile or map object must touch the logical grid at a consistent point. If each crop is exported with a different internal alignment, the renderer will produce visible gaps, floating tiles, or overlap problems.

This script helps prepare assets so they can be used with formulas like:

```python
screen_x = (grid_x - grid_y) * (tile_width // 2)
screen_y = (grid_x + grid_y) * (tile_height // 2)
```

## Output files

After running the extractor, the output directory contains:

- `object_000.png`, `object_001.png`, ...
- `metadata.json`
- `anchors_override.json`
- `_debug/contact_sheet.png`

### metadata.json
Stores information about each extracted object, such as:

- original bounding box
- detected anchor
- paste offset
- output filename

### anchors_override.json
Lets you manually correct bad anchor positions without changing the detection code.

### contact_sheet.png
A visual sheet that shows all exported objects together for quick inspection.

## Intended use

This script is intended for:

- isometric terrain preparation
- RTS map asset extraction
- preprocessing painterly or hand-made isometric tilesets
- Pygame or other 2D engines using a diamond-based isometric grid

## Limitations

This is a heuristic extractor. It works best when:

- the background is visually separable
- objects do not heavily overlap
- each tile-like object is already isolated in the source art

For highly stylized or overlapping art, some anchors may need manual correction using `anchors_override.json`.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Typical workflow

1. Place your source image as `tiles.png`
2. Run the extractor script
3. Review `_debug/contact_sheet.png`
4. Adjust `anchors_override.json` if needed
5. Re-run in reprocess mode if you want corrected exports

## Summary

The main purpose of this script is to convert a raw isometric tileset image into engine-ready transparent objects that align correctly on a diamond-based grid.
