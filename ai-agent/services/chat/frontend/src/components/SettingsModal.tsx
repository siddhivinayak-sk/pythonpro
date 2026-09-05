import { type CSSProperties, useEffect, useState } from "react";
import {
  type ChatSettingsData,
  type ModelInfo,
  getConversationSettings,
  putConversationSettings,
  putSettings,
} from "../api";
import { type Theme, palette } from "../theme";

type Scope = "user" | "conversation";

/** Settings panel for the requirement parameters. Saves at the account (user) scope or, when a
 *  conversation is open, as a per-conversation override (system < user < conversation). RAG collections
 *  and MCP servers are chosen from the deployment's discovered options. */
export function SettingsModal({
  models,
  initial,
  conversationId,
  currentModelRef,
  availableCollections,
  availableServers,
  theme,
  onClose,
  onSaved,
}: {
  models: ModelInfo[];
  initial: ChatSettingsData;
  conversationId: string | null;
  currentModelRef: string;
  availableCollections: string[];
  availableServers: string[];
  theme: Theme;
  onClose: () => void;
  onSaved: (settings: ChatSettingsData, scope: Scope) => void;
}) {
  const pal = palette(theme);
  const [scope, setScope] = useState<Scope>("user");
  const [temperature, setTemperature] = useState(0.7);
  const [memoryWindow, setMemoryWindow] = useState(10);
  const [maxTokens, setMaxTokens] = useState(6553);
  const [topP, setTopP] = useState(0.95);
  const [stop, setStop] = useState("");
  const [frequencyPenalty, setFrequencyPenalty] = useState(0);
  const [presencePenalty, setPresencePenalty] = useState(0);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [themeSetting, setThemeSetting] = useState("system");
  const [modelRef, setModelRef] = useState("");
  const [ragSel, setRagSel] = useState<string[]>([]);
  const [mcpSel, setMcpSel] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function populate(data: ChatSettingsData, useModelFallback: boolean) {
    setTemperature(Number(data.temperature ?? 0.7));
    setMemoryWindow(Number(data.memory_window ?? 12));
    setMaxTokens(Number(data.max_tokens ?? 6553));
    setTopP(Number(data.top_p ?? 0.95));
    setStop((data.stop as string) ?? "");
    setFrequencyPenalty(Number(data.frequency_penalty ?? 0));
    setPresencePenalty(Number(data.presence_penalty ?? 0));
    setSystemPrompt((data.system_prompt as string) ?? "");
    setThemeSetting((data.theme as string) ?? "system");
    // In conversation scope, fall back to the model active in the header so the panel matches the chat
    // window even before a per-conversation override is saved. Account scope shows its true value.
    setModelRef(
      data.connection_id && data.model_name
        ? `${data.connection_id}::${data.model_name}`
        : useModelFallback
          ? currentModelRef
          : "",
    );
    setRagSel(data.rag_collections ?? []);
    setMcpSel(data.mcp_servers ?? []);
  }

  useEffect(() => {
    populate(initial, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function switchScope(next: Scope) {
    setScope(next);
    setErr(null);
    if (next === "conversation" && conversationId) {
      setLoading(true);
      try {
        populate(await getConversationSettings(conversationId), true);
      } catch (e) {
        setErr(String(e));
      } finally {
        setLoading(false);
      }
    } else {
      populate(initial, false);
    }
  }

  function toggle(list: string[], setList: (v: string[]) => void, value: string) {
    setList(list.includes(value) ? list.filter((x) => x !== value) : [...list, value]);
  }

  async function save() {
    setSaving(true);
    setErr(null);
    const [connection_id, model_name] = modelRef ? modelRef.split("::") : [null, null];
    const base: ChatSettingsData = {
      temperature,
      memory_window: memoryWindow,
      max_tokens: maxTokens,
      top_p: topP,
      stop: stop.trim() || null,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      system_prompt: systemPrompt.trim() || null,
      connection_id,
      model_name,
      rag_collections: ragSel,
      mcp_servers: mcpSel,
    };
    try {
      if (scope === "conversation" && conversationId) {
        onSaved(await putConversationSettings(conversationId, base), "conversation");
      } else {
        onSaved(await putSettings({ ...base, theme: themeSetting }), "user");
      }
      onClose();
    } catch (e) {
      setErr(String(e));
    } finally {
      setSaving(false);
    }
  }

  const ragOptions = Array.from(new Set([...availableCollections, ...ragSel]));
  const mcpOptions = Array.from(new Set([...availableServers, ...mcpSel]));
  const defaultModel = models.find((m) => m.is_default);

  const overlay: CSSProperties = {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.45)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 50,
  };
  const panel: CSSProperties = {
    width: "min(560px, 92vw)",
    maxHeight: "88vh",
    overflow: "auto",
    background: pal.appBg,
    color: pal.text,
    border: `1px solid ${pal.border}`,
    borderRadius: 12,
    padding: 20,
    boxShadow: "0 10px 40px rgba(0,0,0,0.3)",
  };
  const label: CSSProperties = { display: "block", fontSize: 12, fontWeight: 600, color: pal.muted, margin: "14px 0 4px" };
  const field: CSSProperties = {
    width: "100%",
    boxSizing: "border-box",
    padding: 8,
    borderRadius: 8,
    border: `1px solid ${pal.inputBorder}`,
    background: pal.inputBg,
    color: pal.text,
    fontSize: 14,
  };
  const btn = (primary: boolean): CSSProperties => ({
    padding: "8px 16px",
    borderRadius: 8,
    border: primary ? "none" : `1px solid ${pal.inputBorder}`,
    background: primary ? pal.accent : "transparent",
    color: primary ? "#fff" : pal.text,
    cursor: "pointer",
  });
  const scopeTab = (active: boolean, disabled: boolean): CSSProperties => ({
    flex: 1,
    padding: "6px 10px",
    borderRadius: 8,
    border: `1px solid ${active ? pal.accent : pal.inputBorder}`,
    background: active ? pal.accent : pal.inputBg,
    color: active ? "#fff" : disabled ? pal.muted : pal.text,
    cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 13,
  });
  const checkList: CSSProperties = {
    border: `1px solid ${pal.inputBorder}`,
    borderRadius: 8,
    padding: 8,
    maxHeight: 120,
    overflow: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  };

  return (
    <div style={overlay} onClick={onClose}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <h2 style={{ margin: "0 0 10px", fontSize: 18 }}>Settings</h2>

        <div style={{ display: "flex", gap: 8 }}>
          <button style={scopeTab(scope === "user", false)} onClick={() => void switchScope("user")}>
            My account
          </button>
          <button
            style={scopeTab(scope === "conversation", !conversationId)}
            disabled={!conversationId}
            title={conversationId ? "" : "Open a conversation to set per-conversation overrides"}
            onClick={() => conversationId && void switchScope("conversation")}
          >
            This conversation
          </button>
        </div>
        <p style={{ margin: "8px 0 0", fontSize: 12, color: pal.muted }}>
          {scope === "user"
            ? "Defaults for your account (system < user < conversation)."
            : "Overrides for this conversation only."}
        </p>

        {loading ? (
          <p style={{ color: pal.muted, marginTop: 16 }}>Loading…</p>
        ) : (
          <>
            <label style={label}>Default model</label>
            <select style={field} value={modelRef} onChange={(e) => setModelRef(e.target.value)}>
              <option value="">
                {defaultModel ? `(server default — ${defaultModel.display_name})` : "(server default)"}
              </option>
              {models.map((m) => (
                <option key={`${m.connection_id}::${m.model_name}`} value={`${m.connection_id}::${m.model_name}`}>
                  {m.display_name} — {m.connection_id}
                </option>
              ))}
            </select>

            <label style={label}>Temperature: {temperature.toFixed(2)}</label>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              style={{ width: "100%" }}
            />

            <label style={label}>Past messages included</label>
            <input
              type="number"
              min={0}
              max={100}
              style={field}
              value={memoryWindow}
              onChange={(e) => setMemoryWindow(Number(e.target.value))}
            />

            <label style={label}>Max completion tokens</label>
            <input
              type="number"
              min={1}
              style={field}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            />

            <label style={label}>Top P: {topP.toFixed(2)}</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
              style={{ width: "100%" }}
            />

            <label style={label}>Frequency penalty: {frequencyPenalty.toFixed(2)}</label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={frequencyPenalty}
              onChange={(e) => setFrequencyPenalty(Number(e.target.value))}
              style={{ width: "100%" }}
            />

            <label style={label}>Presence penalty: {presencePenalty.toFixed(2)}</label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={presencePenalty}
              onChange={(e) => setPresencePenalty(Number(e.target.value))}
              style={{ width: "100%" }}
            />

            <label style={label}>Stop sequence</label>
            <input
              style={field}
              value={stop}
              placeholder="Force-cut output when this string occurs (blank = none)"
              onChange={(e) => setStop(e.target.value)}
            />

            <label style={label}>System instruction</label>
            <textarea
              style={{ ...field, minHeight: 80, resize: "vertical", fontFamily: "inherit" }}
              value={systemPrompt}
              placeholder="Optional persona / instructions applied to every turn"
              onChange={(e) => setSystemPrompt(e.target.value)}
            />

            {scope === "user" && (
              <>
                <label style={label}>Theme</label>
                <select style={field} value={themeSetting} onChange={(e) => setThemeSetting(e.target.value)}>
                  <option value="system">System</option>
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                </select>
              </>
            )}

            <label style={label}>Knowledge (RAG collections)</label>
            {ragOptions.length === 0 ? (
              <p style={{ fontSize: 13, color: pal.muted, margin: 0 }}>No collections available.</p>
            ) : (
              <div style={checkList}>
                {ragOptions.map((name) => (
                  <label key={name} style={{ display: "flex", gap: 8, fontSize: 14, cursor: "pointer" }}>
                    <input type="checkbox" checked={ragSel.includes(name)} onChange={() => toggle(ragSel, setRagSel, name)} />
                    {name}
                  </label>
                ))}
              </div>
            )}

            <label style={label}>Tools (MCP servers)</label>
            {mcpOptions.length === 0 ? (
              <p style={{ fontSize: 13, color: pal.muted, margin: 0 }}>No MCP servers configured.</p>
            ) : (
              <div style={checkList}>
                {mcpOptions.map((url) => (
                  <label key={url} style={{ display: "flex", gap: 8, fontSize: 14, cursor: "pointer" }}>
                    <input type="checkbox" checked={mcpSel.includes(url)} onChange={() => toggle(mcpSel, setMcpSel, url)} />
                    {url}
                  </label>
                ))}
              </div>
            )}
          </>
        )}

        {err && <div style={{ color: "#dc2626", fontSize: 13, marginTop: 12 }}>{err}</div>}

        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 20 }}>
          <button style={btn(false)} onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button style={btn(true)} onClick={() => void save()} disabled={saving || loading}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
