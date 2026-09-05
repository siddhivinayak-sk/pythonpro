// Theming: a small palette keyed by light/dark, plus resolution of the "system" setting.

export type Theme = "light" | "dark";
export type ThemeSetting = "light" | "dark" | "system";

export interface Palette {
  appBg: string;
  text: string;
  sidebarBg: string;
  sidebarText: string;
  border: string;
  bubbleUserBg: string;
  bubbleUserText: string;
  bubbleAssistantBg: string;
  bubbleAssistantText: string;
  inputBg: string;
  inputBorder: string;
  muted: string;
  accent: string;
  codeBg: string;
  codeText: string;
  codeBorder: string;
  link: string;
}

const LIGHT: Palette = {
  appBg: "#ffffff",
  text: "#0f172a",
  sidebarBg: "#0f172a",
  sidebarText: "#e2e8f0",
  border: "#e5e7eb",
  bubbleUserBg: "#4338ca",
  bubbleUserText: "#ffffff",
  bubbleAssistantBg: "#f1f5f9",
  bubbleAssistantText: "#0f172a",
  inputBg: "#ffffff",
  inputBorder: "#cbd5e1",
  muted: "#64748b",
  accent: "#4338ca",
  codeBg: "#f6f8fa",
  codeText: "#1f2937",
  codeBorder: "#e5e7eb",
  link: "#2563eb",
};

const DARK: Palette = {
  appBg: "#0b1220",
  text: "#e2e8f0",
  sidebarBg: "#0f172a",
  sidebarText: "#e2e8f0",
  border: "#1f2a44",
  bubbleUserBg: "#4338ca",
  bubbleUserText: "#ffffff",
  bubbleAssistantBg: "#111827",
  bubbleAssistantText: "#e5e7eb",
  inputBg: "#0f172a",
  inputBorder: "#334155",
  muted: "#94a3b8",
  accent: "#6366f1",
  codeBg: "#0d1117",
  codeText: "#e6edf3",
  codeBorder: "#30363d",
  link: "#60a5fa",
};

export function palette(theme: Theme): Palette {
  return theme === "dark" ? DARK : LIGHT;
}

export function resolveTheme(setting: ThemeSetting): Theme {
  if (setting === "system") {
    const prefersDark =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    return prefersDark ? "dark" : "light";
  }
  return setting;
}
