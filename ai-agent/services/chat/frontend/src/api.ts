// Typed client for the chat backend. Token is kept in localStorage and sent as a Bearer header.

const TOKEN_KEY = "ai-agent-token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}`, ...extra } : extra;
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, { ...init, headers: authHeaders(init.headers as Record<string, string>) });
  if (!res.ok) throw new Error((await res.text()) || `${res.status}`);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

function jsonInit(method: string, body: unknown): RequestInit {
  return { method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
}

// -- types --
export interface ModelInfo {
  connection_id: string;
  model_name: string;
  display_name: string;
  provider: string;
  capabilities: string[];
}
export interface User {
  id: string;
  role: string;
  source: string;
  display_name: string | null;
  email: string | null;
}
export interface Conversation {
  id: string;
  title: string;
  updated_at: string;
  pinned: boolean;
}
export interface Message {
  id: string;
  role: string;
  content: string;
  connection_id: string | null;
  model_name: string | null;
  created_at: string;
}
export interface ChatBody {
  text: string;
  connection_id?: string | null;
  model_name?: string | null;
  temperature?: number | null;
  images?: string[];
}

// -- auth --
export async function login(username: string, password: string): Promise<User> {
  const res = await req<{ token: string; user: User }>("/v1/auth/login", jsonInit("POST", { username, password }));
  setToken(res.token);
  return res.user;
}
export const me = () => req<User>("/v1/auth/me");
export async function logout(): Promise<void> {
  try {
    await req<void>("/v1/auth/logout", { method: "POST" });
  } finally {
    setToken(null);
  }
}

// -- models / conversations --
export const listModels = () => req<ModelInfo[]>("/v1/models");
export const listConversations = () =>
  req<{ conversations: Conversation[] }>("/v1/conversations").then((r) => r.conversations);
export const createConversation = (title?: string) =>
  req<Conversation>("/v1/conversations", jsonInit("POST", { title }));
export const deleteConversation = (id: string) => req<void>(`/v1/conversations/${id}`, { method: "DELETE" });
export const listMessages = (id: string) =>
  req<{ messages: Message[] }>(`/v1/conversations/${id}/messages`).then((r) => r.messages);
export const sendChat = (id: string, body: ChatBody) =>
  req<Message>(`/v1/conversations/${id}/chat`, jsonInit("POST", body));

// -- settings / uploads --
export interface ChatSettingsData {
  temperature?: number;
  memory_window?: number; // "past messages included"
  max_tokens?: number;
  top_p?: number;
  stop?: string | null;
  frequency_penalty?: number;
  presence_penalty?: number;
  system_prompt?: string | null;
  theme?: string; // light | dark | system
  connection_id?: string | null;
  model_name?: string | null;
  rag_collections?: string[];
  mcp_servers?: string[];
  [key: string]: unknown;
}
export const getSettings = () =>
  req<{ settings: ChatSettingsData }>("/v1/settings").then((r) => r.settings);
export const putSettings = (data: ChatSettingsData) =>
  req<{ settings: ChatSettingsData }>("/v1/settings", jsonInit("PUT", { data })).then((r) => r.settings);
export const getConversationSettings = (convId: string) =>
  req<{ settings: ChatSettingsData }>(`/v1/conversations/${convId}/settings`).then((r) => r.settings);
export const putConversationSettings = (convId: string, data: ChatSettingsData) =>
  req<{ settings: ChatSettingsData }>(`/v1/conversations/${convId}/settings`, jsonInit("PUT", { data })).then(
    (r) => r.settings,
  );

// -- discovery (for the settings UI) --
export const listRagCollections = () =>
  req<{ collections: string[] }>("/v1/rag/collections").then((r) => r.collections);
export const listMcpServers = () => req<{ servers: string[] }>("/v1/mcp/servers").then((r) => r.servers);

export interface ContextFile {
  id: string;
  filename: string;
  chars: number;
}
export function uploadContext(convId: string, file: File): Promise<ContextFile> {
  const form = new FormData();
  form.append("file", file);
  return fetch(`/v1/conversations/${convId}/context`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  }).then((r) => {
    if (!r.ok) throw new Error(`context upload failed: ${r.status}`);
    return r.json() as Promise<ContextFile>;
  });
}
export const listContext = (convId: string) =>
  req<{ context: ContextFile[] }>(`/v1/conversations/${convId}/context`).then((r) => r.context);

// -- streaming chat (SSE over fetch) --
export async function streamChat(id: string, body: ChatBody, onDelta: (t: string) => void): Promise<void> {
  const res = await fetch(`/v1/conversations/${id}/chat/stream`, {
    ...jsonInit("POST", body),
    headers: authHeaders({ "content-type": "application/json" }),
  });
  if (!res.ok || !res.body) throw new Error(`stream failed: ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const evt of events) {
      const line = evt.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        const obj = JSON.parse(payload) as { delta?: string; error?: string };
        if (obj.error) throw new Error(obj.error);
        if (obj.delta) onDelta(obj.delta);
      } catch {
        /* ignore keep-alive / partial frames */
      }
    }
  }
}
