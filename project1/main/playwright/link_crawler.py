"""Crawl a (possibly authenticated) documentation site with Playwright and report link status as JSON.

First run opens a visible browser so you can log in; the session is saved and reused on later runs.

pip install -r requirements.txt
python -m playwright install chromium

Examples:
    python link_crawler.py https://docs.example.com/
    python link_crawler.py https://docs.example.com/guide/ --scope https://docs.example.com/guide/ -o report.json
    python link_crawler.py https://docs.example.com/ --login        # force re-authentication
"""

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

SKIP_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "about:")

# Collects every <a href> with its text, a context snippet and an XPath to locate it later.
EXTRACT_LINKS_JS = """
(contextChars) => {
  const xpath = (el) => {
    const parts = [];
    for (; el && el.nodeType === 1; el = el.parentNode) {
      let i = 1;
      for (let s = el.previousElementSibling; s; s = s.previousElementSibling)
        if (s.nodeName === el.nodeName) i++;
      parts.unshift(el.nodeName.toLowerCase() + '[' + i + ']');
    }
    return '/' + parts.join('/');
  };
  const blockTags = new Set(['P','LI','TD','TH','DD','DT','H1','H2','H3','H4','H5','H6',
                             'PRE','BLOCKQUOTE','SECTION','ARTICLE','NAV','DIV','MAIN','BODY']);
  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
  return Array.from(document.querySelectorAll('a[href]')).map((a, idx) => {
    let block = a.parentElement;
    while (block && !blockTags.has(block.tagName)) block = block.parentElement;
    const blockText = norm(block ? block.innerText : '');
    const text = norm(a.innerText) || norm(a.getAttribute('aria-label')) || norm(a.title);
    let before = '', after = '';
    const pos = text ? blockText.indexOf(text) : -1;
    if (pos >= 0) {
      before = blockText.slice(Math.max(0, pos - contextChars), pos);
      after = blockText.slice(pos + text.length, pos + text.length + contextChars);
    }
    let heading = '';
    const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'));
    for (const h of headings) {
      if (h.compareDocumentPosition(a) & Node.DOCUMENT_POSITION_FOLLOWING) heading = norm(h.innerText);
      else break;
    }
    return {
      index: idx,
      text,
      href: a.getAttribute('href'),
      resolved: a.href,
      context_before: before,
      context_after: after,
      nearest_heading: heading,
      xpath: xpath(a),
    };
  });
}
"""


def normalize(url: str) -> str:
    return urldefrag(url)[0]


def in_scope(url: str, scope: str) -> bool:
    u, s = urlparse(url), urlparse(scope)
    return u.scheme in ("http", "https") and u.netloc == s.netloc and u.path.startswith(s.path)


def authenticate(pw, start_url: str, state_file: Path) -> None:
    browser = pw.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(start_url)
    print("\nA browser window has opened. Complete the login, wait until the docs page is shown,")
    input("then press ENTER here to save the session... ")
    context.storage_state(path=str(state_file))
    browser.close()
    print(f"Session saved to {state_file}")


def check_url(request_ctx, url: str, timeout_ms: int) -> dict:
    try:
        resp = request_ctx.head(url, timeout=timeout_ms, fail_on_status_code=False)
        # Many servers reject/misreport HEAD; confirm with GET.
        if resp.status >= 400 or resp.status == 0:
            resp = request_ctx.get(url, timeout=timeout_ms, fail_on_status_code=False)
        return {"status": resp.status, "final_url": resp.url, "error": None}
    except PlaywrightError as e:
        return {"status": None, "final_url": None, "error": str(e).splitlines()[0]}


def crawl(args) -> None:
    start_url = normalize(args.url)
    scope = args.scope or start_url.rsplit("/", 1)[0] + "/"
    state_file = Path(args.auth_state)

    with sync_playwright() as pw:
        if args.login or not state_file.exists():
            authenticate(pw, start_url, state_file)

        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(storage_state=str(state_file), ignore_https_errors=args.ignore_https_errors)
        page = context.new_page()

        queue = deque([start_url])
        visited: set[str] = set()
        page_status: dict[str, dict] = {}
        raw_links: list[dict] = []

        while queue and len(visited) < args.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            print(f"[{len(visited)}] {url}")
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=args.timeout * 1000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightError:
                    pass
                status = resp.status if resp else None
                page_status[url] = {"status": status, "final_url": page.url, "error": None}
                ctype = (resp.headers.get("content-type", "") if resp else "")
                if (status and status >= 400) or "html" not in ctype:
                    continue
                if not in_scope(normalize(page.url), scope):
                    continue  # redirected out of scope (e.g. login page => session expired)
                links = page.evaluate(EXTRACT_LINKS_JS, args.context_chars)
            except PlaywrightError as e:
                page_status[url] = {"status": None, "final_url": None, "error": str(e).splitlines()[0]}
                continue

            for link in links:
                href = (link["href"] or "").strip()
                if not href or href.lower().startswith(SKIP_SCHEMES) or (href.startswith("#") and not args.include_anchors):
                    continue
                target = link["resolved"] or urljoin(url, href)
                link["source_page"] = url
                link["target"] = target
                raw_links.append(link)
                target_norm = normalize(target)
                if in_scope(target_norm, scope) and target_norm not in visited:
                    queue.append(target_norm)
            if args.delay:
                time.sleep(args.delay)

        # Resolve status for every unique target (reusing crawled page results).
        status_cache: dict[str, dict] = dict(page_status)
        unique_targets = {normalize(l["target"]) for l in raw_links} - status_cache.keys()
        print(f"Checking {len(unique_targets)} additional target URLs...")
        for i, target in enumerate(sorted(unique_targets), 1):
            status_cache[target] = check_url(context.request, target, args.timeout * 1000)
            print(f"  ({i}/{len(unique_targets)}) {status_cache[target]['status'] or status_cache[target]['error']}  {target}")

        browser.close()

    results = []
    for l in raw_links:
        info = status_cache.get(normalize(l["target"]), {})
        status = info.get("status")
        results.append({
            "source_page": l["source_page"],
            "link_text": l["text"],
            "target_link": l["target"],
            "href_attribute": l["href"],
            "error_code": status,
            "error_message": info.get("error"),
            "final_url": info.get("final_url"),
            "is_broken": status is None or not 200 <= status < 400,
            "surrounding_text": {
                "before": l["context_before"],
                "after": l["context_after"],
                "nearest_heading": l["nearest_heading"],
            },
            "xpath": l["xpath"],
            "link_index_on_page": l["index"],
        })

    failed = [r for r in results if r["is_broken"]]
    if args.only_broken:
        results = failed

    failures_by_code: dict[str, int] = {}
    for r in failed:
        key = str(r["error_code"]) if r["error_code"] is not None else "request_failed"
        failures_by_code[key] = failures_by_code.get(key, 0) + 1

    report = {
        "start_url": start_url,
        "scope": scope,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "pages_crawled": len(visited),
            "links_found": len(raw_links),
            "failed_links": len(failed),
            "unique_failed_targets": len({normalize(r["target_link"]) for r in failed}),
            "failures_by_code": failures_by_code,
        },
        "failed_links": failed,
        "links": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    print(f"\nPages: {s['pages_crawled']}  Links: {s['links_found']}  Failed: {s['failed_links']}  By code: {failures_by_code}")
    print(f"Report written to {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Crawl an authenticated docs site and validate links.")
    p.add_argument("url", help="Start URL")
    p.add_argument("--scope", help="Only crawl URLs starting with this prefix (default: start URL's directory)")
    p.add_argument("-o", "--output", default=f"link_report_{datetime.now():%Y%m%d_%H%M%S}.json")
    p.add_argument("--auth-state", default="auth_state.json", help="Saved session file (contains cookies - keep private)")
    p.add_argument("--login", action="store_true", help="Force interactive login even if a session exists")
    p.add_argument("--max-pages", type=int, default=1000)
    p.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds")
    p.add_argument("--delay", type=float, default=0, help="Seconds to wait between page visits")
    p.add_argument("--context-chars", type=int, default=60, help="Chars of surrounding text before/after link")
    p.add_argument("--include-anchors", action="store_true", help="Include same-page '#anchor' links")
    p.add_argument("--only-broken", action="store_true", help="Write only broken links to the report")
    p.add_argument("--headed", action="store_true", help="Show the browser while crawling")
    p.add_argument("--ignore-https-errors", action="store_true")
    args = p.parse_args()
    try:
        crawl(args)
    except KeyboardInterrupt:
        sys.exit("Interrupted.")


if __name__ == "__main__":
    main()
