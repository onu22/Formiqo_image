## Same goal
Both versions answer: **where should each form field be stamped on the page?**

They both use **200 DPI page images** and **OpenCV line detection** as the form’s skeleton.

---

## How you run it

**Before harness (this branch)** — you drive it in steps:
1. Convert + detect lines  
2. Run AI grounding  
3. Stamp values  

**Harness-built** — upload once, and the app runs the full chain for you (QA still optional/off by default).

---

## What happens after lines are detected

Think of it as: **who draws the final box?**

### Before harness
```
Lines detected
    ↓
AI looks at page + lines
    ↓
AI says: “Field goes in this pixel box {x,y,w,h}”
    ↓
App snaps that box onto nearby cell/band if it fits
    ↓
Save fields
```

The AI is mainly a **box drawer**. Lines help it, and snapping cleans up bad guesses.

### Harness-built
```
Lines detected → build cells & writable bands
    ↓
AI looks at page + lines
    ↓
AI says: “This field belongs to these lines / this cell / this label”
    ↓
App computes the real box from that structure
    ↓
(only if needed) fall back to AI’s pixel guess + snap
    ↓
Save fields
    ↓
(optional) Vision QA: stamp → judge crop → tiny nudge
```

The AI is mainly a **matcher** (“which structure is this field?”). Geometry draws most of the final boxes.

---

## Side-by-side

| | Before harness | Harness-built |
|---|---|---|
| **Pipeline** | Manual 3 API steps | Auto after upload |
| **AI’s job** | Guess pixel boxes | Propose anchors (which lines/cells) |
| **Final boxes** | Mostly AI + snap | Mostly geometry from anchors |
| **Lines’ role** | Guide + snap target | Source of truth for stamp regions |
| **Vision QA** | None | Optional second pass (usually off) |
| **Template reuse** | None | Can skip AI if form seen before |

---

## One-sentence difference

**Before harness:** AI draws boxes, app lightly corrects them.  
**Harness-built:** AI points at form structure, app draws the boxes — then optionally checks them visually.
