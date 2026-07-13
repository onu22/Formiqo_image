Coordinate grid overlay (pixel-fallback aid)

When enabled (`FORMIQO_GROUNDING_GRID_OVERLAY_ENABLED`), the highlighted page image is drawn
with a labeled coordinate grid before it is sent to the model:

- Thin gridlines every `grounding_grid_overlay_spacing_px` pixels (default 100).
- Each gridline is labeled with its pixel coordinate along the top edge (x) and left edge (y).
- Origin is top-left, matching the page metadata JSON coordinate system.

Use the grid only to improve raw pixel estimates for fields that cannot be anchored to a
detected line or a printed label. Anchors (cell / line_anchor / label_anchor) remain
preferred and authoritative; the grid is a fallback for unanchored fields.

The overlay is cosmetic: it does not change page dimensions and must not be treated as
document content.
