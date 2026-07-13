import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
  {
    variants: {
      variant: {
        neutral: "bg-slate-100 text-slate-700 border-slate-200",
        success: "bg-green-50 text-green-700 border-green-200",
        info: "bg-blue-50 text-blue-700 border-blue-200",
        brand: "bg-brand-50 text-brand-700 border-brand-200",
        danger: "bg-red-50 text-red-700 border-red-200",
        warning: "bg-amber-50 text-amber-700 border-amber-200",
      },
    },
    defaultVariants: {
      variant: "neutral",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
