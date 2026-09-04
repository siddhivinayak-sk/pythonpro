import { type FormEvent, useState } from "react";
import { login, type User } from "../api";

export function Login({ onLogin }: { onLogin: (u: User) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onLogin(await login(username, password));
    } catch {
      setError("Login failed — check your credentials.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={s.page}>
      <form style={s.card} onSubmit={submit}>
        <h1 style={s.title}>AI-Agent</h1>
        <p style={s.sub}>Sign in to continue</p>
        <label style={s.label}>Username</label>
        <input style={s.input} value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
        <label style={s.label}>Password</label>
        <input style={s.input} type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
        {error && <p style={s.error}>{error}</p>}
        <button style={s.button} disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        <p style={s.hint}>Admin sign-in, or configure Keycloak/OAuth2 (see docs).</p>
      </form>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  page: { minHeight: "100vh", display: "grid", placeItems: "center", fontFamily: "system-ui, sans-serif", background: "#f8fafc", margin: 0 },
  card: { width: 340, padding: 28, background: "white", borderRadius: 14, boxShadow: "0 10px 30px rgba(0,0,0,.08)", display: "flex", flexDirection: "column" },
  title: { margin: "0 0 4px", fontSize: 22 },
  sub: { margin: "0 0 18px", color: "#64748b", fontSize: 14 },
  label: { fontSize: 12, fontWeight: 600, color: "#334155", margin: "8px 0 4px" },
  input: { padding: 10, borderRadius: 8, border: "1px solid #cbd5e1", fontSize: 14 },
  button: { marginTop: 16, padding: "10px 14px", borderRadius: 8, border: "none", background: "#4338ca", color: "white", fontSize: 14, cursor: "pointer" },
  error: { color: "#b91c1c", fontSize: 13, marginTop: 10 },
  hint: { color: "#94a3b8", fontSize: 12, marginTop: 14 },
};
