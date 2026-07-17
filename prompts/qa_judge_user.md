Review the placement of this single field and return your verdict via the structured output.

Field context (positions are in original page pixels, top-left origin):

- `field_id`, `type`, printed `label`
- `value`: the text/mark stamped onto the form
- `bbox`: the field's current bounding box `{x, y, w, h}` in page pixels
- `page`: the page `{width, height}` in pixels
- `crop_zoom`: how much the attached crop is magnified vs. the page

The attached image is the zoom crop for this field, with its bbox drawn as a debug rectangle.

Return:

- `verdict`: `ok` or `adjust`
- `dx`, `dy`: page-pixel correction (0/0 when `ok`)
- `confidence`: `[0, 1]`
- `reason`: one short phrase (e.g. "value floats above line", "sits correctly on line")
