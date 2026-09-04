import { useEffect, useState } from "react";
import { getToken, me, type User } from "./api";
import { Chat } from "./components/Chat";
import { Login } from "./components/Login";

/**
 * Top-level auth gate: shows the login screen until a valid session exists, then the chat workspace.
 * (Phase 3 UI: login, conversation history, streaming chat, model switching, and image paste/vision.)
 */
export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!getToken()) {
      setLoading(false);
      return;
    }
    me()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 24, fontFamily: "system-ui" }}>Loading…</div>;
  if (!user) return <Login onLogin={setUser} />;
  return <Chat user={user} onLogout={() => setUser(null)} />;
}
