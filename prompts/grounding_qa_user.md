Review the placement of exactly one field.

You are given:
1. field_metadata_json — the field_id, type, the stamped value, and the field's current bbox
   in full-page pixel coordinates (top-left origin), plus the crop's offset and scale so you
   can translate what you see back into page pixels.
2. field_crop_image — a zoomed crop of the rendered page centered on this field. A debug
   rectangle marks the field's current bounding box.

Decide whether the stamped value is correctly positioned on its writing line or inside its
cell/box. If it is correct, return verdict `ok` with dx = 0 and dy = 0. If it is off, return
verdict `shift` with the pixel translation (dx right positive, dy down positive) that would
move the field onto its correct line or into its correct cell.

Answer for this field only.
