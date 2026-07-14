import { FileText } from "lucide-react";
import { Link } from "react-router-dom";

export function FormiqoLogo({ onBeforeNavigate }: { onBeforeNavigate?: () => boolean }) {
  return (
    <Link
      to="/"
      className="flex items-center gap-2 text-lg font-semibold text-slate-900"
      onClick={(e) => {
        if (onBeforeNavigate && !onBeforeNavigate()) e.preventDefault();
      }}
    >
      <FileText className="h-6 w-6 text-brand-600" strokeWidth={2.25} />
      Formiqo
    </Link>
  );
}
