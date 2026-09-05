import type { CSSProperties, ReactNode } from "react";
import { type Theme, palette } from "../theme";

// A small, dependency-free markdown renderer that returns React elements (never dangerouslySetInnerHTML,
// so text is auto-escaped by React). URLs/image sources are sanitized to block javascript:/other schemes.
// Supports: fenced code, headings, blockquotes, hr, ordered/unordered lists, GFM pipe tables, paragraphs,
// and inline bold/italic/code/links/images (incl. bare image URLs and data-URIs).

function safeHref(url: string): string {
  return /^(https?:|mailto:|\/|#)/i.test(url.trim()) ? url : "#";
}
function safeImg(src: string): string {
  return /^(https?:|data:image\/)/i.test(src.trim()) ? src : "";
}
function isImageUrl(url: string): boolean {
  return /\.(png|jpe?g|gif|webp|svg)(\?\S*)?$/i.test(url);
}

const INLINE_RE = new RegExp(
  [
    "!\\[([^\\]]*)\\]\\(([^)\\s]+)\\)", // 1 alt, 2 img src
    "\\[([^\\]]+)\\]\\(([^)\\s]+)\\)", // 3 text, 4 href
    "`([^`]+)`", // 5 inline code
    "\\*\\*([^*]+)\\*\\*", // 6 bold
    "__([^_]+)__", // 7 bold
    "\\*([^*]+)\\*", // 8 italic
    "(data:image\\/[A-Za-z]+;base64,[A-Za-z0-9+/=]+)", // 9 data-uri image
    "(https?:\\/\\/[^\\s)]+)", // 10 bare url
  ].join("|"),
  "g",
);

function renderInline(text: string, pal: ReturnType<typeof palette>, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = new RegExp(INLINE_RE.source, "g");
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  const inlineCode: CSSProperties = {
    background: pal.codeBg,
    border: `1px solid ${pal.codeBorder}`,
    borderRadius: 4,
    padding: "1px 5px",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "0.9em",
  };
  const imgStyle: CSSProperties = { maxWidth: "100%", borderRadius: 8, display: "block", margin: "6px 0" };
  const linkStyle: CSSProperties = { color: pal.link, textDecoration: "underline" };
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const k = `${keyBase}-${key++}`;
    if (m[2] !== undefined) {
      const src = safeImg(m[2]);
      if (src) out.push(<img key={k} src={src} alt={m[1]} style={imgStyle} />);
    } else if (m[4] !== undefined) {
      out.push(
        <a key={k} href={safeHref(m[4])} target="_blank" rel="noreferrer noopener" style={linkStyle}>
          {m[3]}
        </a>,
      );
    } else if (m[5] !== undefined) {
      out.push(
        <code key={k} style={inlineCode}>
          {m[5]}
        </code>,
      );
    } else if (m[6] !== undefined) {
      out.push(<strong key={k}>{m[6]}</strong>);
    } else if (m[7] !== undefined) {
      out.push(<strong key={k}>{m[7]}</strong>);
    } else if (m[8] !== undefined) {
      out.push(<em key={k}>{m[8]}</em>);
    } else if (m[9] !== undefined) {
      out.push(<img key={k} src={m[9]} alt="" style={imgStyle} />);
    } else if (m[10] !== undefined) {
      const url = m[10];
      if (isImageUrl(url)) out.push(<img key={k} src={safeImg(url)} alt="" style={imgStyle} />);
      else
        out.push(
          <a key={k} href={safeHref(url)} target="_blank" rel="noreferrer noopener" style={linkStyle}>
            {url}
          </a>,
        );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function renderMultiline(text: string, pal: ReturnType<typeof palette>, keyBase: string): ReactNode[] {
  const lines = text.split("\n");
  const out: ReactNode[] = [];
  lines.forEach((line, idx) => {
    if (idx > 0) out.push(<br key={`${keyBase}-br-${idx}`} />);
    out.push(...renderInline(line, pal, `${keyBase}-l${idx}`));
  });
  return out;
}

function isBlockStart(line: string): boolean {
  return (
    /^```/.test(line) ||
    /^(#{1,6})\s+/.test(line) ||
    /^(-{3,}|\*{3,}|_{3,})\s*$/.test(line) ||
    /^\s*>/.test(line) ||
    /^(\s*)([-*+]|\d+\.)\s+/.test(line)
  );
}

function splitRow(row: string): string[] {
  return row
    .replace(/^\s*\|/, "")
    .replace(/\|\s*$/, "")
    .split("|")
    .map((c) => c.trim());
}

export function Markdown({ text, theme }: { text: string; theme: Theme }): ReactNode {
  const pal = palette(theme);
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;

  const cellStyle: CSSProperties = { border: `1px solid ${pal.codeBorder}`, padding: "4px 8px", textAlign: "left" };

  while (i < lines.length) {
    const line = lines[i];

    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        body.push(lines[i]);
        i++;
      }
      i++; // consume closing fence
      blocks.push(
        <pre
          key={key++}
          style={{
            background: pal.codeBg,
            color: pal.codeText,
            border: `1px solid ${pal.codeBorder}`,
            borderRadius: 8,
            padding: 12,
            overflowX: "auto",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            fontSize: 13,
            margin: "8px 0",
          }}
        >
          <code>{body.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    if (/^\s*$/.test(line)) {
      i++;
      continue;
    }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      const level = Math.min(h[1].length, 6);
      const Tag = `h${level}` as keyof JSX.IntrinsicElements;
      blocks.push(
        <Tag key={key++} style={{ margin: "10px 0 6px", lineHeight: 1.3, fontSize: `${1.5 - (level - 1) * 0.12}em` }}>
          {renderInline(h[2], pal, `h${key}`)}
        </Tag>,
      );
      i++;
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      blocks.push(<hr key={key++} style={{ border: "none", borderTop: `1px solid ${pal.border}`, margin: "10px 0" }} />);
      i++;
      continue;
    }

    if (/^\s*>/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        body.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      blocks.push(
        <blockquote
          key={key++}
          style={{ borderLeft: `3px solid ${pal.border}`, margin: "8px 0", padding: "2px 12px", color: pal.muted }}
        >
          {renderMultiline(body.join("\n"), pal, `bq${key}`)}
        </blockquote>,
      );
      continue;
    }

    // GFM pipe table: header row + a separator row of dashes
    if (
      line.includes("|") &&
      i + 1 < lines.length &&
      /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) &&
      lines[i + 1].includes("-")
    ) {
      const header = splitRow(line);
      let j = i + 2;
      const rows: string[][] = [];
      while (j < lines.length && lines[j].includes("|") && !/^\s*$/.test(lines[j])) {
        rows.push(splitRow(lines[j]));
        j++;
      }
      blocks.push(
        <table key={key++} style={{ borderCollapse: "collapse", margin: "8px 0", fontSize: 14 }}>
          <thead>
            <tr>
              {header.map((cell, x) => (
                <th key={x} style={{ ...cellStyle, background: pal.codeBg }}>
                  {renderInline(cell, pal, `th${key}-${x}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, y) => (
              <tr key={y}>
                {row.map((cell, x) => (
                  <td key={x} style={cellStyle}>
                    {renderInline(cell, pal, `td${key}-${y}-${x}`)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>,
      );
      i = j;
      continue;
    }

    const listMatch = line.match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
    if (listMatch) {
      const ordered = /\d+\./.test(listMatch[2]);
      const items: ReactNode[] = [];
      while (i < lines.length) {
        const li = lines[i].match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
        if (!li) break;
        items.push(<li key={items.length}>{renderInline(li[3], pal, `li${key}-${items.length}`)}</li>);
        i++;
      }
      const ListTag = ordered ? "ol" : "ul";
      blocks.push(
        <ListTag key={key++} style={{ margin: "6px 0", paddingLeft: 22 }}>
          {items}
        </ListTag>,
      );
      continue;
    }

    // paragraph
    const para: string[] = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !isBlockStart(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={key++} style={{ margin: "6px 0" }}>
        {renderMultiline(para.join("\n"), pal, `p${key}`)}
      </p>,
    );
  }

  return <div style={{ lineHeight: 1.55 }}>{blocks}</div>;
}
