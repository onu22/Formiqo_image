import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { JobStatus } from "@/lib/types";

const STATUS_CONFIG: Record<
  JobStatus,
  { label: string; variant: "success" | "info" | "brand" | "danger"; icon: "check" | "spin" | "warn" }
> = {
  ready: { label: "Ready for review", variant: "success", icon: "check" },
  exported: { label: "Exported", variant: "brand", icon: "check" },
  converting: { label: "Converting...", variant: "info", icon: "spin" },
  grounding: { label: "Grounding fields...", variant: "info", icon: "spin" },
  failed: { label: "Failed", variant: "danger", icon: "warn" },
};

export function StatusBadge({ status }: { status: JobStatus }) {
  const config = STATUS_CONFIG[status] ?? { label: status, variant: "neutral" as const, icon: "check" as const };
  return (
    <Badge variant={config.variant}>
      {config.icon === "check" && <CheckCircle2 className="h-3.5 w-3.5" />}
      {config.icon === "spin" && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {config.icon === "warn" && <AlertTriangle className="h-3.5 w-3.5" />}
      {config.label}
    </Badge>
  );
}
