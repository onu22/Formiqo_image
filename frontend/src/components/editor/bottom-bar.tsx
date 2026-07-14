import { useEditorStore } from "@/store/editorStore";
import { shortId } from "@/lib/utils";

export function EditorBottomBar() {
  const { fieldsById, jobId } = useEditorStore();
  const fields = Object.values(fieldsById);
  const reviewedCount = fields.filter((f) => f.reviewed).length;

  return (
    <footer className="flex h-9 items-center justify-between border-t border-slate-200 bg-white px-4 text-xs text-slate-500">
      <span>
        {reviewedCount} of {fields.length} fields reviewed
      </span>
      <span>
        Job {jobId ? shortId(jobId) : "\u2014"} &bull; Grounded with AI vision
      </span>
    </footer>
  );
}
