import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500/40",
  {
    variants: {
      variant: {
        primary: "bg-brand-600 text-white hover:bg-brand-700 shadow-sm",
        outline: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
        outlineBrand: "border border-brand-300 bg-white text-brand-700 hover:bg-brand-50",
        ghost: "text-slate-600 hover:bg-slate-100",
        link: "text-brand-600 hover:underline underline-offset-2 font-medium",
        danger: "bg-red-600 text-white hover:bg-red-700",
        dangerGhost: "text-red-600 hover:bg-red-50",
      },
      size: {
        default: "h-9 px-4",
        sm: "h-8 px-3 text-sm",
        xs: "h-7 px-2 text-xs",
        icon: "h-9 w-9",
        iconSm: "h-7 w-7",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, type = "button", ...props }, ref) => {
    return (
      <button
        ref={ref}
        type={type}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";
