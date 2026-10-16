"""Crawl a locally generated HTML documentation folder with Playwright and report link status as JSON.

Starts at index.html, follows meta-refresh redirects and <a href> links to every local HTML page,
checks local files exist on disk and (optionally) external http(s) links respond.

Examples:
    python local_link_crawler.py C:\\path\\to\\generated_html\\index.html
    python local_link_crawler.py C:\\path\\to\\generated_html --skip-external --check-anchors -o report.json
"""

import argparse
import json
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import url2pathname

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from link_crawler import EXTRACT_LINKS_JS, SKIP_SCHEMES, check_url, normalize

META_REFRESH_RE = re.compile(
    r"""<meta[^>]+http-equiv=["']?refresh["']?[^>]*content=["']?\s*\d+\s*;\s*url=([^"'>\s]+)""", re.I
)
COLLECT_IDS_JS = "() => Array.from(document.querySelectorAll('[id], a[name]')).map(e => e.id || e.getAttribute('name'))"


def to_path(url: str) -> Path:
    return Path(url2pathname(urlparse(url).path))


def is_crawlable(url: str, root: Path) -> bool:
    if urlparse(url).scheme != "file":
        return False
    p = to_path(url)
    return p.suffix.lower() in (".html", ".htm") and p.is_file() and p.resolve().is_relative_to(root)


def check_local(url: str, allow_dir_index: bool) -> dict:
    p = to_path(url)
    if p.is_file():
        return {"status": 200, "final_url": url, "error": None}
    if p.is_dir():
        index = p / "index.html"
        if index.is_file():
            if allow_dir_index:
                return {"status": 200, "final_url": index.as_uri(), "error": None}
            return {"status": 404, "final_url": None, "error": "Directory link; file:// does not open index.html"}
        return {"status": 404, "final_url": None, "error": "Directory link without index.html"}
    return {"status": 404, "final_url": None, "error": f"File not found: {p}"}


def crawl(args) -> None:
    start = Path(args.index).resolve()
    if start.is_dir():
        start = start / "index.html"
    if not start.is_file():
        sys.exit(f"Start file not found: {start}")
    root = Path(args.root).resolve() if args.root else start.parent
    start_url = start.as_uri()
    include_anchors = args.include_anchors or args.check_anchors

    queue = deque([start_url])
    visited: set[str] = set()
    page_ids: dict[str, set[str]] = {}
    raw_links: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_context().new_page()

        while queue and len(visited) < args.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            print(f"[{len(visited)}] {to_path(url).relative_to(root)}")

            # Redirect-only pages (e.g. root index.html) navigate away immediately; handle them without the browser.
            html = to_path(url).read_text(encoding="utf-8", errors="ignore")
            meta = META_REFRESH_RE.search(html)
            if meta:
                href = meta.group(1)
                links = [{
                    "index": 0, "text": "(meta refresh)", "href": href, "resolved": urljoin(url, href),
                    "context_before": "", "context_after": "", "nearest_heading": "",
                    "xpath": "/html/head/meta[@http-equiv='refresh']",
                }]
            else:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=args.timeout * 1000)
                    links = page.evaluate(EXTRACT_LINKS_JS, args.context_chars)
                    page_ids[url] = set(page.evaluate(COLLECT_IDS_JS))
                except PlaywrightError as e:
                    print(f"    ! {str(e).splitlines()[0]}")
                    continue

            for link in links:
                href = (link["href"] or "").strip()
                if not href or href.lower().startswith(SKIP_SCHEMES) or (href.startswith("#") and not include_anchors):
                    continue
                target = link["resolved"] or urljoin(url, href)
                link["source_page"] = url
                link["target"] = target
                raw_links.append(link)
                target_norm = normalize(target)
                if target_norm not in visited and is_crawlable(target_norm, root):
                    queue.append(target_norm)

        status_cache: dict[str, dict] = {}
        request_ctx = pw.request.new_context(ignore_https_errors=args.ignore_https_errors)
        unique_targets = sorted({normalize(l["target"]) for l in raw_links})
        print(f"Checking {len(unique_targets)} unique target URLs...")
        for target in unique_targets:
            scheme = urlparse(target).scheme
            if scheme == "file":
                status_cache[target] = check_local(target, args.allow_dir_index)
            elif scheme in ("http", "https"):
                if args.skip_external:
                    status_cache[target] = {"status": None, "final_url": None, "error": "not checked", "skipped": True}
                else:
                    status_cache[target] = check_url(request_ctx, target, args.timeout * 1000)
                    print(f"  {status_cache[target]['status'] or status_cache[target]['error']}  {target}")
            else:
                status_cache[target] = {"status": None, "final_url": None, "error": f"Unsupported scheme: {scheme}"}
        request_ctx.dispose()
        browser.close()

    results = []
    for l in raw_links:
        target_norm, fragment = urldefrag(l["target"])
        info = status_cache.get(target_norm, {})
        status = info.get("status")
        error = info.get("error")
        skipped = info.get("skipped", False)
        broken = not skipped and (status is None or not 200 <= status < 400)
        if args.check_anchors and not broken and fragment and target_norm in page_ids:
            if unquote(fragment) not in page_ids[target_norm]:
                broken, error = True, f"Anchor '#{fragment}' not found on target page"
        results.append({
            "source_page": l["source_page"],
            "source_file": str(to_path(l["source_page"])),
            "link_text": l["text"],
            "target_link": l["target"],
            "href_attribute": l["href"],
            "error_code": status,
            "error_message": error,
            "final_url": info.get("final_url"),
            "is_broken": broken,
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
        if r["error_message"] and r["error_message"].startswith("Anchor"):
            key = "anchor_missing"
        failures_by_code[key] = failures_by_code.get(key, 0) + 1

    report = {
        "start_file": str(start),
        "root": str(root),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "pages_crawled": len(visited),
            "links_found": len(raw_links),
            "failed_links": len(failed),
            "unique_failed_targets": len({r["target_link"] for r in failed}),
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
    p = argparse.ArgumentParser(description="Crawl local HTML docs and validate links.")
    p.add_argument("index", help="Path to index.html (or the folder containing it)")
    p.add_argument("--root", help="Only crawl HTML files under this folder (default: folder of index.html)")
    p.add_argument("-o", "--output", default=f"local_link_report_{datetime.now():%Y%m%d_%H%M%S}.json")
    p.add_argument("--max-pages", type=int, default=5000)
    p.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds")
    p.add_argument("--context-chars", type=int, default=60, help="Chars of surrounding text before/after link")
    p.add_argument("--include-anchors", action="store_true", help="Include same-page '#anchor' links")
    p.add_argument("--check-anchors", action="store_true", help="Verify '#fragment' targets exist on local pages")
    p.add_argument("--allow-dir-index", action="store_true",
                   help="Treat links to folders containing index.html as valid (as a web server would)")
    p.add_argument("--skip-external", action="store_true", help="Do not check http(s) links")
    p.add_argument("--only-broken", action="store_true", help="Write only broken links to 'links'")
    p.add_argument("--headed", action="store_true", help="Show the browser while crawling")
    p.add_argument("--ignore-https-errors", action="store_true")
    args = p.parse_args()
    try:
        crawl(args)
    except KeyboardInterrupt:
        sys.exit("Interrupted.")


if __name__ == "__main__":
    main()
