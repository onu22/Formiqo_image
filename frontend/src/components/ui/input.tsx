import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          "h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900 placeholder:text-slate-400 outline-none transition-colors focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 disabled:bg-slate-50 disabled:text-slate-400",
          className,
        )}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";
