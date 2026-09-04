import { type ChangeEvent, type ClipboardEvent, type ReactNode, useEffect, useRef, useState } from "react";
import {
  type ContextFile,
  type Conversation,
  type ModelInfo,
  type User,
  createConversation,
  deleteConversation,
  listConversations,
  listContext,
  listMessages,
  listModels,
  logout as apiLogout,
  streamChat,
  uploadContext,
} from "../api";

type UIMessage = { role: string; content: string; model_name?: string | null };

// Match markdown images, bare http(s) image URLs, and inline data-URI images so assistant
// output that references an image renders inline in the transcript.
const IMG_RE =
  /!\[[^\]]*\]\(([^)]+)\)|(https?:\/\/\S+\.(?:png|jpe?g|gif|webp|svg))|(data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+)/g;

function renderContent(text: string): ReactNode {
  const nodes: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  IMG_RE.lastIndex = 0;
  while ((m = IMG_RE.exec(text)) !== null) {
    if (m.index > last) nodes.push(<span key={key++}>{text.slice(last, m.index)}</span>);
    const src = m[1] || m[2] || m[3];
    nodes.push(<img key={key++} src={src} alt="" style={s.outImg} />);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(<span key={key++}>{text.slice(last)}</span>);
  return nodes.length ? nodes : text;
}

export function Chat({ user, onLogout }: { user: User; onLogout: () => void }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedRef, setSelectedRef] = useState<string>("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contextFiles, setContextFiles] = useState<ContextFile[]>([]);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const contextInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        if (m.length) setSelectedRef(`${m[0].connection_id}::${m[0].model_name}`);
      })
      .catch((e) => setError(String(e)));
    refreshConversations();
  }, []);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight });
  }, [messages]);

  function refreshConversations() {
    listConversations().then(setConversations).catch(() => undefined);
  }

  function refreshContext(id: string) {
    listContext(id).then(setContextFiles).catch(() => setContextFiles([]));
  }

  async function openConversation(id: string) {
    setActiveId(id);
    setMessages((await listMessages(id)).map((m) => ({ role: m.role, content: m.content, model_name: m.model_name })));
    refreshContext(id);
  }

  async function newChat() {
    const conv = await createConversation();
    setConversations((c) => [conv, ...c]);
    setActiveId(conv.id);
    setMessages([]);
    setContextFiles([]);
  }

  async function onContextSelected(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    let convId = activeId;
    if (!convId) {
      const conv = await createConversation();
      setConversations((c) => [conv, ...c]);
      convId = conv.id;
      setActiveId(conv.id);
    }
    try {
      await uploadContext(convId, file);
      refreshContext(convId);
    } catch (err) {
      setError(String(err));
    }
  }

  async function removeConversation(id: string) {
    await deleteConversation(id);
    setConversations((c) => c.filter((x) => x.id !== id));
    if (activeId === id) {
      setActiveId(null);
      setMessages([]);
      setContextFiles([]);
    }
  }

  function onPaste(e: ClipboardEvent<HTMLTextAreaElement>) {
    for (const item of Array.from(e.clipboardData.items)) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (!file) continue;
        const reader = new FileReader();
        reader.onload = () => setImages((imgs) => [...imgs, reader.result as string]);
        reader.readAsDataURL(file);
      }
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || streaming) return;
    let convId = activeId;
    if (!convId) {
      const conv = await createConversation();
      setConversations((c) => [conv, ...c]);
      convId = conv.id;
      setActiveId(conv.id);
    }
    const [connection_id, model_name] = selectedRef.split("::");
    const body = { text, connection_id, model_name, images: images.length ? images : undefined };
    setInput("");
    setImages([]);
    setError(null);
    setMessages((m) => [...m, { role: "user", content: text }, { role: "assistant", content: "", model_name }]);
    setStreaming(true);
    try {
      await streamChat(convId, body, (delta) =>
        setMessages((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          copy[copy.length - 1] = { ...last, content: last.content + delta };
          return copy;
        }),
      );
      refreshConversations();
    } catch (e) {
      setError(String(e));
    } finally {
      setStreaming(false);
    }
  }

  async function doLogout() {
    await apiLogout();
    onLogout();
  }

  return (
    <div style={s.app}>
      <aside style={s.sidebar}>
        <button style={s.newBtn} onClick={newChat}>+ New chat</button>
        <div style={s.convList}>
          {conversations.map((c) => (
            <div key={c.id} style={{ ...s.convItem, ...(c.id === activeId ? s.convActive : {}) }}>
              <span style={s.convTitle} onClick={() => openConversation(c.id)}>{c.title}</span>
              <button style={s.convDel} title="Delete" onClick={() => removeConversation(c.id)}>×</button>
            </div>
          ))}
        </div>
        <div style={s.userBox}>
          <span>{user.display_name ?? user.email ?? "user"} ({user.role})</span>
          <button style={s.logout} onClick={doLogout}>Sign out</button>
        </div>
      </aside>

      <main style={s.main}>
        <header style={s.header}>
          <label style={s.modelLabel}>Model</label>
          <select style={s.modelSelect} value={selectedRef} onChange={(e) => setSelectedRef(e.target.value)}>
            {models.map((m) => (
              <option key={`${m.connection_id}::${m.model_name}`} value={`${m.connection_id}::${m.model_name}`}>
                {m.display_name} — {m.connection_id}
              </option>
            ))}
          </select>
        </header>

        <div style={s.transcript} ref={transcriptRef}>
          {messages.length === 0 && <p style={s.hint}>Start a conversation. Paste an image to use vision-capable models.</p>}
          {messages.map((m, i) => (
            <div key={i} style={{ ...s.msg, ...(m.role === "user" ? s.msgUser : s.msgAssistant) }}>
              <div style={s.msgRole}>{m.role}{m.model_name ? ` · ${m.model_name}` : ""}</div>
              <div style={s.msgContent}>
                {m.content
                  ? m.role === "assistant"
                    ? renderContent(m.content)
                    : m.content
                  : streaming && i === messages.length - 1
                    ? "…"
                    : ""}
              </div>
            </div>
          ))}
        </div>

        {error && <div style={s.errorBar}>{error}</div>}

        <div style={s.composer}>
          {images.length > 0 && (
            <div style={s.thumbs}>
              {images.map((src, i) => (
                <img key={i} src={src} alt="pasted" style={s.thumb} onClick={() => setImages((im) => im.filter((_, j) => j !== i))} />
              ))}
            </div>
          )}
          <div style={s.contextRow}>
            <input
              ref={contextInputRef}
              type="file"
              accept=".txt,.md,.markdown,.json,.csv,.log,text/*"
              style={{ display: "none" }}
              onChange={(e) => void onContextSelected(e)}
            />
            <button style={s.contextBtn} title="Attach a text file as reference context" onClick={() => contextInputRef.current?.click()}>
              📎 Add context
            </button>
            {contextFiles.length > 0 && (
              <span style={s.contextInfo}>
                {contextFiles.length} file{contextFiles.length === 1 ? "" : "s"}: {contextFiles.map((c) => c.filename).join(", ")}
              </span>
            )}
          </div>
          <textarea
            style={s.input}
            value={input}
            placeholder="Message… (Enter to send, Shift+Enter for newline; paste an image for vision)"
            onChange={(e) => setInput(e.target.value)}
            onPaste={onPaste}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <button style={s.send} disabled={streaming} onClick={() => void send()}>{streaming ? "…" : "Send"}</button>
        </div>
      </main>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  app: { display: "flex", height: "100vh", fontFamily: "system-ui, sans-serif", margin: 0 },
  sidebar: { width: 260, background: "#0f172a", color: "#e2e8f0", display: "flex", flexDirection: "column", padding: 12 },
  newBtn: { padding: "10px 12px", borderRadius: 8, border: "1px solid #334155", background: "#1e293b", color: "white", cursor: "pointer" },
  convList: { flex: 1, overflow: "auto", marginTop: 12 },
  convItem: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 10px", borderRadius: 8, cursor: "pointer" },
  convActive: { background: "#1e293b" },
  convTitle: { flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 13 },
  convDel: { background: "transparent", border: "none", color: "#64748b", cursor: "pointer", fontSize: 16 },
  userBox: { borderTop: "1px solid #334155", paddingTop: 10, fontSize: 12, display: "flex", flexDirection: "column", gap: 6 },
  logout: { background: "transparent", border: "1px solid #334155", color: "#e2e8f0", borderRadius: 6, padding: "4px 8px", cursor: "pointer" },
  main: { flex: 1, display: "flex", flexDirection: "column", minWidth: 0 },
  header: { display: "flex", alignItems: "center", gap: 8, padding: 12, borderBottom: "1px solid #e5e7eb" },
  modelLabel: { fontSize: 12, fontWeight: 600, color: "#475569" },
  modelSelect: { padding: 6, borderRadius: 8, border: "1px solid #cbd5e1" },
  transcript: { flex: 1, overflow: "auto", padding: 24, display: "flex", flexDirection: "column", gap: 14 },
  hint: { color: "#94a3b8", fontSize: 14 },
  msg: { maxWidth: 720, padding: "10px 14px", borderRadius: 12 },
  msgUser: { alignSelf: "flex-end", background: "#4338ca", color: "white" },
  msgAssistant: { alignSelf: "flex-start", background: "#f1f5f9", color: "#0f172a" },
  msgRole: { fontSize: 11, opacity: 0.7, marginBottom: 4 },
  msgContent: { whiteSpace: "pre-wrap", lineHeight: 1.5 },
  outImg: { maxWidth: "100%", borderRadius: 8, display: "block", margin: "6px 0" },
  contextRow: { display: "flex", alignItems: "center", gap: 8, width: "100%", flexWrap: "wrap" },
  contextBtn: { padding: "6px 10px", borderRadius: 8, border: "1px solid #cbd5e1", background: "#f8fafc", color: "#334155", cursor: "pointer", fontSize: 13 },
  contextInfo: { fontSize: 12, color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 480 },
  errorBar: { background: "#fee2e2", color: "#b91c1c", padding: "8px 16px", fontSize: 13 },
  composer: { display: "flex", gap: 8, padding: 16, borderTop: "1px solid #e5e7eb", alignItems: "flex-end", flexWrap: "wrap" },
  thumbs: { display: "flex", gap: 6, width: "100%" },
  thumb: { height: 48, borderRadius: 6, cursor: "pointer", border: "1px solid #cbd5e1" },
  input: { flex: 1, minHeight: 44, maxHeight: 160, padding: 10, borderRadius: 8, border: "1px solid #cbd5e1", resize: "vertical", fontFamily: "inherit", fontSize: 14 },
  send: { padding: "10px 18px", borderRadius: 8, border: "none", background: "#4338ca", color: "white", cursor: "pointer" },
};
