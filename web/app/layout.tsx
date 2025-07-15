import type { Metadata } from "next";
import "./globals.css";
import "@mantine/core/styles.css";

import { Inter } from "next/font/google";

import { createTheme, MantineProvider } from "@mantine/core";

import { ConvexClientProvider } from "./ConvexClientProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "iSE Mancala",
  description: "by Logan Fossenier",
};

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
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </head>
      <body className={inter.className}>
        <MantineProvider defaultColorScheme="dark" theme={theme}>
          <ConvexClientProvider>{children}</ConvexClientProvider>
        </MantineProvider>
      </body>
    </html>
  );
}
