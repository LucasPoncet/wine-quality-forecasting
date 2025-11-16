import type { Metadata } from "next";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });
import "./globals.css";

export const metadata: Metadata = {
  title: "Wine Quality Forecasting | ML Project",
  description: "redicting French wine vintage quality using 10+ years of weather data, deep learning models, and geospatial preprocessing.",
   icons: {
    icon: "/icon.svg",                 
    shortcut: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={inter.className}
      >
        {children}
      </body>
    </html>
  );
}
