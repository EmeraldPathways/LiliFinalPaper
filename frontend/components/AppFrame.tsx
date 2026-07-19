"use client";

import type { ReactNode } from "react";

import { usePathname } from "next/navigation";

import { TopNav } from "@/components/TopNav";

type AppFrameProps = {
  children: ReactNode;
};

export function AppFrame({ children }: AppFrameProps) {
  const pathname = usePathname();
  const isArtifactRoute = pathname === "/artifact-demo";

  if (isArtifactRoute) {
    return (
      <div className="artifact-route-shell">
        <div className="artifact-route-container">{children}</div>
      </div>
    );
  }

  return (
    <div className="shell">
      <div className="container">
        <header className="hero">
          <span className="eyebrow">Offline E-commerce Experiment</span>
          <h1>Agentic AI Recommendation Framework</h1>
          <p>
            A research demo comparing collaborative filtering against an explainable,
            feedback-aware recommendation decision layer built on the H&amp;M dataset.
          </p>
          <TopNav />
        </header>
        {children}
      </div>
    </div>
  );
}
