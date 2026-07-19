import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppFrame } from "@/components/AppFrame";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Agentic AI Recommendation Demo",
  description: "Academic dashboard comparing collaborative filtering and agentic AI recommendations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AppFrame>{children}</AppFrame>
      </body>
    </html>
  );
}
