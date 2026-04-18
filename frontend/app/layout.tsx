import type { Metadata } from "next";
import "./globals.css";
import CommandBar from "@/components/CommandBar";

export const metadata: Metadata = {
  title: "PulseIQ",
  description: "Near real-time Economic Stress Intelligence for US geographies",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen flex flex-col" suppressHydrationWarning>
        <CommandBar />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
