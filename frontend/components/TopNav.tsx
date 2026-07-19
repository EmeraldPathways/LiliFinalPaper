"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/artifact-demo", "Artifact Demo"],
  ["/", "Overview"],
  ["/research-setup", "Research Setup"],
  ["/data-processing", "Data Processing"],
  ["/user-intention", "User Intention"],
  ["/comparison", "Comparison"],
  ["/evaluation", "Evaluation"],
  ["/explainability-evidence", "Explainability Evidence"],
] as const;

export function TopNav() {
  const pathname = usePathname();
  const currentLabel = links.find(([href]) => href === pathname)?.[1] ?? "Overview";

  return (
    <>
      <div className="nav-status">
        <span className="eyebrow">Current Page</span>
        <strong>{currentLabel}</strong>
      </div>
      <nav className="nav" aria-label="Primary">
        {links.map(([href, label]) => {
          const isActive = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              aria-current={isActive ? "page" : undefined}
              className={isActive ? "nav-link nav-link-active" : "nav-link"}
            >
              {label}
            </Link>
          );
        })}
      </nav>
    </>
  );
}
