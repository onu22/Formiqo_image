import {
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  CheckCircle2,
  Minus,
  Plus,
  RotateCcw,
} from "lucide-react";
import { useEditorStore } from "@/store/editorStore";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const TYPE_LABELS: Record<string, string> = {
  text: "text",
  multiline_text: "multiline",
  checkbox: "checkbox",
  radio: "radio",
};

export function Inspector() {
  const store = useEditorStore();
  const { fieldsById, fieldOrderByPage, currentPage, values, selectedFieldId } = store;
  const field = selectedFieldId ? fieldsById[selectedFieldId] : null;
  const fieldIds = fieldOrderByPage[currentPage] ?? [];

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-800">Field Inspector</h2>
      </div>

      <div className="flex-1 overflow-y-auto">
        {field ? (
          <div className="space-y-5 border-b border-slate-200 px-4 py-4">
            <div>
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-slate-500">Field name</label>
                <Badge variant="neutral">{TYPE_LABELS[field.type] ?? field.type}</Badge>
              </div>
              <p className="mt-1 text-sm font-semibold text-slate-800">{field.field_id}</p>
              {field.label && <p className="text-xs text-slate-400">{field.label}</p>}
            </div>

            {field.qa_status === "flagged" && (
              <div
                data-testid="inspector-flagged"
                className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800"
              >
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                <span>
                  QA flagged this field &mdash; the vision check could not confirm its placement.
                  Please verify its position on the page.
                </span>
              </div>
            )}

            {field.type === "checkbox" || field.type === "radio" ? (
              <div>
                <label className="text-xs font-medium text-slate-500">Value</label>
                <label className="mt-2 flex items-center gap-2 text-sm text-slate-700">
                  <input
                    type="checkbox"
                    checked={["true", "1", "yes", "on", "checked", "x"].includes(
                      (values[field.field_id] ?? "").trim().toLowerCase(),
                    )}
                    onChange={(e) => store.updateValue(field.field_id, e.target.checked ? "true" : "")}
                    className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                  />
                  Checked
                </label>
              </div>
            ) : (
              <div>
                <label className="text-xs font-medium text-slate-500">Value</label>
                {field.type === "multiline_text" ? (
                  <textarea
                    data-testid="inspector-value"
                    value={values[field.field_id] ?? ""}
                    onChange={(e) => store.updateValue(field.field_id, e.target.value)}
                    rows={3}
                    className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20"
                  />
                ) : (
                  <Input
                    data-testid="inspector-value"
                    className="mt-1"
                    value={values[field.field_id] ?? ""}
                    onChange={(e) => store.updateValue(field.field_id, e.target.value)}
                  />
                )}
              </div>
            )}

            <div>
              <label className="text-xs font-medium text-slate-500">Font size</label>
              <div className="mt-1 flex items-center gap-2">
                <Button
                  data-testid="font-size-minus"
                  variant="outline"
                  size="iconSm"
                  onClick={() => store.updateFontSize(field.field_id, (field.font_size_pt ?? 11) - 1)}
                >
                  <Minus className="h-3.5 w-3.5" />
                </Button>
                <span className="w-16 text-center text-sm text-slate-700">{field.font_size_pt ?? 11} pt</span>
                <Button
                  data-testid="font-size-plus"
                  variant="outline"
                  size="iconSm"
                  onClick={() => store.updateFontSize(field.field_id, (field.font_size_pt ?? 11) + 1)}
                >
                  <Plus className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>

            <div>
              <label className="text-xs font-medium text-slate-500">Position</label>
              <div className="mt-1 grid grid-cols-2 gap-3">
                <div>
                  <span className="text-[11px] text-slate-400">X</span>
                  <Input
                    data-testid="inspector-x"
                    type="number"
                    value={Math.round(field.bbox.x)}
                    onChange={(e) =>
                      store.setBBoxLive(field.field_id, { ...field.bbox, x: Number(e.target.value) || 0 })
                    }
                  />
                </div>
                <div>
                  <span className="text-[11px] text-slate-400">Y</span>
                  <Input
                    data-testid="inspector-y"
                    type="number"
                    value={Math.round(field.bbox.y)}
                    onChange={(e) =>
                      store.setBBoxLive(field.field_id, { ...field.bbox, y: Number(e.target.value) || 0 })
                    }
                  />
                </div>
              </div>

              <div className="mt-3 flex items-center justify-between">
                <span className="text-[11px] text-slate-400">Nudge (1px)</span>
                <div className="grid grid-cols-3 gap-1">
                  <span />
                  <Button variant="outline" size="iconSm" onClick={() => store.nudgeField(field.field_id, 0, -1)}>
                    <ArrowUp className="h-3.5 w-3.5" />
                  </Button>
                  <span />
                  <Button variant="outline" size="iconSm" onClick={() => store.nudgeField(field.field_id, -1, 0)}>
                    <ArrowLeft className="h-3.5 w-3.5" />
                  </Button>
                  <span />
                  <Button variant="outline" size="iconSm" onClick={() => store.nudgeField(field.field_id, 1, 0)}>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Button>
                  <span />
                  <Button variant="outline" size="iconSm" onClick={() => store.nudgeField(field.field_id, 0, 1)}>
                    <ArrowDown className="h-3.5 w-3.5" />
                  </Button>
                  <span />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <Button variant="outline" onClick={() => store.resetPosition(field.field_id)}>
                <RotateCcw className="h-3.5 w-3.5" />
                Reset position
              </Button>
              <Button
                variant={field.reviewed ? "outlineBrand" : "outline"}
                onClick={() => store.toggleReviewed(field.field_id, !field.reviewed)}
              >
                <CheckCircle2 className="h-3.5 w-3.5" />
                {field.reviewed ? "Reviewed" : "Mark reviewed"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="px-4 py-8 text-center text-sm text-slate-400">
            Select a field on the page to edit its value, position, and size.
          </div>
        )}

        <div className="px-4 py-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Fields on this page ({fieldIds.length})
          </h3>
          <ul className="mt-3 space-y-1">
            {fieldIds.map((fieldId) => {
              const f = fieldsById[fieldId];
              if (!f) return null;
              const isCheckboxLike = f.type === "checkbox" || f.type === "radio";
              const value = values[fieldId] ?? "";
              const preview = isCheckboxLike
                ? ["true", "1", "yes", "on", "checked", "x"].includes(value.trim().toLowerCase())
                  ? "checked"
                  : "unchecked"
                : value;
              return (
                <li key={fieldId}>
                  <button
                    data-testid={`field-list-${fieldId}`}
                    onClick={() => store.selectField(fieldId)}
                    className={cn(
                      "flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
                      selectedFieldId === fieldId ? "bg-brand-50 text-brand-800" : "hover:bg-slate-50",
                    )}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium text-slate-700">{fieldId}</span>
                      <span className="block truncate text-xs text-slate-400">{preview || "\u2014"}</span>
                    </span>
                    {f.qa_status === "flagged" && (
                      <AlertTriangle
                        data-testid={`field-list-flagged-${fieldId}`}
                        className="h-4 w-4 shrink-0 text-amber-500"
                        aria-label="QA flagged"
                      />
                    )}
                    {f.reviewed ? (
                      <CheckCircle2 className="h-4 w-4 shrink-0 text-green-500" />
                    ) : (
                      <span className="h-4 w-4 shrink-0 rounded-full border-2 border-slate-300" />
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </aside>
  );
}
