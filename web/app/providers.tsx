"use client";

import { ConvexProvider, ConvexReactClient } from "convex/react";
import { ReactNode } from "react";

import { ConvexAuthNextjsProvider } from "@convex-dev/auth/nextjs";
import { createTheme, MantineProvider } from "@mantine/core";

const convex = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL!);

const theme = createTheme({
  // Must have at least 10 colours
  colors: {
    example_blue: [
      "#e3f2fd",
      "#bbdefb",
      "#90caf9",
      "#64b5f6",
      "#42a5f5",
      "#2196f3",
      "#1e88e5",
      "#1976d2",
      "#1565c0",
      "#0d47a1",
    ],
    example_magenta: [
      "#fce4ec",
      "#f8bbd0",
      "#f48fb1",
      "#f06292",
      "#ec407a",
      "#e91e63",
      "#d81b60",
      "#c2185b",
      "#ad1457",
      "#880e4f",
    ],
  },
  primaryShade: { light: 6, dark: 7 },
  fontFamily: "Inter, sans-serif",
  fontSizes: {
    xs: "1.2rem",
    sm: "1.4rem",
    md: "1.6rem",
    lg: "1.8rem",
    xl: "2rem",
  },
});

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ConvexProvider client={convex}>
      <ConvexAuthNextjsProvider>
        <MantineProvider theme={theme}>{children}</MantineProvider>
      </ConvexAuthNextjsProvider>
    </ConvexProvider>
  );
}
