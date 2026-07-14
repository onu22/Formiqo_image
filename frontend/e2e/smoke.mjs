// Manual Playwright smoke script (not part of `npm run build`).
// Exercises the six E6 mockup states against the real FastAPI backend.
import { chromium } from "playwright";

const BASE = process.env.E2E_BASE_URL || "http://localhost:8000";
const READY_JOB = process.env.E2E_READY_JOB;
const FAILED_JOB = process.env.E2E_FAILED_JOB;
const PROCESSING_JOB = process.env.E2E_PROCESSING_JOB;
const SHOT_DIR = "/tmp/pw-screens";

function assert(cond, msg) {
  if (!cond) throw new Error(`ASSERTION FAILED: ${msg}`);
  console.log(`  ok: ${msg}`);
}

async function main() {
  const browser = await chromium.launch({ args: ["--no-sandbox", "--disable-dev-shm-usage"] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });

  console.log("1. Jobs list");
  await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Documents");
  await page.screenshot({ path: `${SHOT_DIR}/1-jobs-list.png`, fullPage: true });
  assert(await page.locator("text=Ready for review").count() >= 1, "jobs list shows Ready for review badge");
  assert((await page.locator("text=Failed").count()) >= 1, "jobs list shows Failed badge");

  console.log("2. Upload page");
  await page.goto(`${BASE}/upload`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Upload a PDF form");
  await page.screenshot({ path: `${SHOT_DIR}/2-upload-page.png`, fullPage: true });

  console.log("3. Processing state");
  await page.goto(`${BASE}/jobs/${PROCESSING_JOB}`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Preparing your form");
  await page.screenshot({ path: `${SHOT_DIR}/3-processing-state.png`, fullPage: true });
  assert(await page.locator("text=Detecting form fields with AI").count() === 1, "processing checklist rendered");

  console.log("4. Failed state");
  await page.goto(`${BASE}/jobs/${FAILED_JOB}`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=We couldn't process this form");
  await page.screenshot({ path: `${SHOT_DIR}/4-failed-state.png`, fullPage: true });
  assert(await page.locator("text=Retry failed pages").count() === 1, "failed state has retry button");

  console.log("5. Editor - load, select, canvas");
  await page.goto(`${BASE}/jobs/${READY_JOB}`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Field Inspector");
  await page.waitForSelector("img[alt='Page 1']");
  await page.screenshot({ path: `${SHOT_DIR}/5a-editor-initial.png`, fullPage: true });

  // Select the date_of_birth field via the "Fields on this page" list.
  await page.click("[data-testid='field-list-date_of_birth']");
  await page.waitForSelector("text=Nudge (1px)");
  const xInput = page.locator("[data-testid='inspector-x']");
  const yInput = page.locator("[data-testid='inspector-y']");
  const xBefore = Number(await xInput.inputValue());
  const yBefore = Number(await yInput.inputValue());
  console.log(`  selected field position before nudge: x=${xBefore} y=${yBefore}`);
  await page.screenshot({ path: `${SHOT_DIR}/5b-editor-selected.png`, fullPage: true });

  console.log("6. Keyboard nudge: 1px arrow, then Shift+10px");
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction(
    (expected) => {
      const el = document.querySelector("[data-testid='inspector-x']");
      return el && Number(el.value) === expected;
    },
    xBefore + 1,
  );
  const xAfterArrow = Number(await xInput.inputValue());
  assert(xAfterArrow === xBefore + 1, `1px ArrowRight moved x from ${xBefore} to ${xAfterArrow}`);

  await page.keyboard.press("Shift+ArrowDown");
  await page.waitForFunction(
    (expected) => {
      const el = document.querySelector("[data-testid='inspector-y']");
      return el && Number(el.value) === expected;
    },
    yBefore + 10,
  );
  const yAfterShift = Number(await yInput.inputValue());
  assert(yAfterShift === yBefore + 10, `Shift+ArrowDown moved y from ${yBefore} to ${yAfterShift}`);

  assert((await page.locator("text=Unsaved changes").count()) === 1, "unsaved-changes indicator shown after nudge");

  console.log("7. Drag the field on canvas");
  const overlay = page.locator("[data-testid='field-overlay-date_of_birth']");
  const box = await overlay.boundingBox();
  assert(box !== null, "selected field overlay has a bounding box");
  const startX = box.x + box.width / 2;
  const startY = box.y + box.height / 2;
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX + 40, startY + 25, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(150);
  const xAfterDrag = Number(await xInput.inputValue());
  const yAfterDrag = Number(await yInput.inputValue());
  console.log(`  position after drag: x=${xAfterDrag} y=${yAfterDrag}`);
  assert(xAfterDrag !== xAfterArrow || yAfterDrag !== yBefore + 10, "drag changed field position");

  console.log("8. Change value and font size");
  const fontSizeText = await page.locator("text=/\\d+ pt/").first().innerText();
  const fontBefore = Number(fontSizeText.replace(" pt", ""));
  const valueInput = page.locator("[data-testid='inspector-value']");
  await valueInput.fill("03/15/1988");
  await page.click("[data-testid='font-size-plus']");

  console.log("9. Save");
  await page.click("button:has-text('Save')");
  await page.waitForFunction(() => !document.body.innerText.includes("Unsaved changes"), null, { timeout: 5000 });
  assert(true, "Save cleared the unsaved-changes indicator");
  await page.screenshot({ path: `${SHOT_DIR}/5c-editor-after-save.png`, fullPage: true });

  console.log("10. Verify server round-trip via API");
  const fieldsRes = await page.evaluate(async (jobId) => {
    const res = await fetch(`/api/v1/jobs/${jobId}/fields`);
    return res.json();
  }, READY_JOB);
  const serverField = fieldsRes.pages[0].fields.find((f) => f.field_id === "date_of_birth");
  console.log("  server bbox:", serverField.bbox, "server value:", fieldsRes.values.date_of_birth);
  assert(serverField.bbox.x === xAfterDrag, `server x ${serverField.bbox.x} matches UI x ${xAfterDrag}`);
  assert(serverField.bbox.y === yAfterDrag, `server y ${serverField.bbox.y} matches UI y ${yAfterDrag}`);
  assert(fieldsRes.values.date_of_birth === "03/15/1988", "server value matches edited value");
  assert(
    serverField.font_size_pt === fontBefore + 1,
    `server font_size_pt is ${fontBefore + 1}, got ${serverField.font_size_pt}`,
  );

  console.log("11. Refresh Preview");
  await page.click("button:has-text('Refresh Preview')");
  await page.waitForResponse((r) => r.url().includes("/stamp-images") && r.status() === 200);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${SHOT_DIR}/5d-editor-after-refresh.png`, fullPage: true });
  assert(true, "stamp-images call completed");

  console.log("12. Export PDF -> success modal");
  await page.click("button:has-text('Export PDF')");
  await page.waitForResponse((r) => r.url().includes("/stamp-pdf") && r.status() === 200);
  await page.waitForSelector("text=Your PDF is ready");
  await page.screenshot({ path: `${SHOT_DIR}/6-export-success.png`, fullPage: true });
  assert(true, "export success modal shown");

  await browser.close();

  if (errors.length > 0) {
    console.error("Console/page errors encountered:");
    for (const e of errors) console.error(" -", e);
    process.exitCode = 1;
  } else {
    console.log("\nAll smoke checks passed with zero console errors.");
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
