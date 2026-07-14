import { useRef } from "react";
import { AlertTriangle, Move } from "lucide-react";
import type { GroundedField } from "@/lib/types";
import { useEditorStore } from "@/store/editorStore";
import { cn } from "@/lib/utils";

const CHECK_TRUE_VALUES = new Set(["true", "1", "yes", "on", "checked", "x"]);

function isTruthyCheckboxValue(value: string | undefined): boolean {
  if (!value) return false;
  return CHECK_TRUE_VALUES.has(value.trim().toLowerCase());
}

export function FieldOverlay({
  field,
  value,
  scale,
  selected,
  showValueText = true,
}: {
  field: GroundedField;
  value: string;
  scale: number;
  selected: boolean;
  /** False when displaying the server-stamped preview, which already bakes the value into the image. */
  showValueText?: boolean;
}) {
  const store = useEditorStore();
  const dragRef = useRef<{
    startX: number;
    startY: number;
    origin: { x: number; y: number; w: number; h: number };
    began: boolean;
  } | null>(null);

  function onPointerMove(e: PointerEvent) {
    const drag = dragRef.current;
    if (!drag) return;
    const dxScreen = e.clientX - drag.startX;
    const dyScreen = e.clientY - drag.startY;
    if (!drag.began && Math.hypot(dxScreen, dyScreen) < 3) return;
    if (!drag.began) {
      store.beginDrag(field.field_id);
      drag.began = true;
    }
    const dxImg = dxScreen / scale;
    const dyImg = dyScreen / scale;
    store.setBBoxLive(field.field_id, {
      ...drag.origin,
      x: drag.origin.x + dxImg,
      y: drag.origin.y + dyImg,
    });
  }

  function onPointerUp() {
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    dragRef.current = null;
  }

  function onPointerDown(e: React.PointerEvent) {
    e.stopPropagation();
    store.selectField(field.field_id);
    dragRef.current = { startX: e.clientX, startY: e.clientY, origin: { ...field.bbox }, began: false };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }

  const isCheckboxLike = field.type === "checkbox" || field.type === "radio";
  const flagged = field.qa_status === "flagged";

  const style: React.CSSProperties = {
    left: field.bbox.x * scale,
    top: field.bbox.y * scale,
    width: field.bbox.w * scale,
    height: field.bbox.h * scale,
  };

  return (
    <div
      data-testid={`field-overlay-${field.field_id}`}
      className={cn(
        "absolute cursor-move select-none",
        selected ? "z-20" : "z-10",
      )}
      style={style}
      onPointerDown={onPointerDown}
    >
      {selected && (
        <span className="absolute -top-6 left-0 flex items-center gap-1 whitespace-nowrap rounded-md bg-brand-600 px-2 py-0.5 text-[10px] font-medium text-white shadow">
          {field.field_id}
        </span>
      )}

      <div
        className={cn(
          "flex h-full w-full items-center overflow-hidden rounded-[2px] border",
          isCheckboxLike ? "justify-center" : "px-1",
          selected
            ? "border-2 border-brand-600 bg-brand-500/10"
            : !showValueText
              ? "border border-transparent hover:border-brand-400/60 hover:bg-brand-400/5"
              : flagged
                ? "border border-amber-400 bg-amber-400/10"
                : "border border-brand-300/70 bg-brand-400/10 hover:border-brand-500",
        )}
      >
        {showValueText &&
          (isCheckboxLike ? (
            isTruthyCheckboxValue(value) && (
              <span className="text-brand-700" style={{ fontSize: Math.max(field.bbox.h * scale * 0.8, 10) }}>
                &#10003;
              </span>
            )
          ) : (
            <span
              className="truncate text-slate-800"
              style={{ fontSize: Math.max((field.font_size_pt ?? 11) * scale * 1.05, 8) }}
            >
              {value}
            </span>
          ))}
      </div>

      {flagged && (
        <span
          className="absolute -right-2 -top-2 flex h-4 w-4 items-center justify-center rounded-full bg-amber-500 text-white"
          title="Check this field"
        >
          <AlertTriangle className="h-2.5 w-2.5" />
        </span>
      )}

      {selected && (
        <span className="absolute -bottom-2 -right-2 flex h-4 w-4 items-center justify-center rounded-full bg-brand-600 text-white shadow">
          <Move className="h-2.5 w-2.5" />
        </span>
      )}
    </div>
  );
}
