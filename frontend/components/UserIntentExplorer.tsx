"use client";

import { useEffect, useState } from "react";

import type { UserIntent } from "@/lib/api";
import { getUserIntent } from "@/lib/api";
import { UserIntentCard } from "@/components/UserIntentCard";

type UserIntentExplorerProps = {
  userIds: string[];
};

export function UserIntentExplorer({ userIds }: UserIntentExplorerProps) {
  const [selectedUser, setSelectedUser] = useState(userIds[0] ?? "");
  const [intent, setIntent] = useState<UserIntent | null>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (!selectedUser) {
      return;
    }
    getUserIntent(selectedUser)
      .then((data) => {
        setIntent(data);
        setError("");
      })
      .catch((fetchError) => {
        setIntent(null);
        setError(fetchError instanceof Error ? fetchError.message : "Unable to fetch user intent.");
      });
  }, [selectedUser]);

  if (!userIds.length) {
    return <section className="card muted">Run the backend experiment to populate sample users.</section>;
  }

  return (
    <div className="stack">
      <div className="controls">
        <select value={selectedUser} onChange={(event) => setSelectedUser(event.target.value)}>
          {userIds.map((userId) => (
            <option key={userId} value={userId}>
              {userId}
            </option>
          ))}
        </select>
      </div>
      {error ? <section className="card muted">{error}</section> : null}
      {intent ? <UserIntentCard intent={intent} /> : null}
    </div>
  );
}

