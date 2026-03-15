import requests
import feedparser
import re
from bs4 import BeautifulSoup
import sys
import json
from datetime import datetime
from dateutil import parser as date_parser


# ===============================
# Dynatrace Configuration
# ===============================
VENDOR_CONFIG = {
    "name": "Dynatrace",
    "base_url": "https://www.dynatrace.com",
    "rss_url": "https://www.dynatrace.com/news/security-alert/feed/",
    "bulletin_url": "https://www.dynatrace.com/news/security-alert/",
    "link_pattern": "/news/security-alert/"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def check_rss(cve_id):
    """
    Check RSS feed for CVE - returns the most recent matching entry.
    Search strategy:
    1. First search titles for CVE ID
    2. If not found in titles, search summaries
    """
    feed = feedparser.parse(VENDOR_CONFIG["rss_url"])
    
    def build_result(entry, found_in):
        """Build result dict from RSS entry."""
        title = entry.title if hasattr(entry, 'title') else ""
        summary = entry.summary if hasattr(entry, 'summary') else ""
        
        # Parse publish date for sorting
        pub_date = None
        if hasattr(entry, 'published'):
            try:
                pub_date = date_parser.parse(entry.published)
            except:
                pass
        
        severity, score = extract_cvss(summary + " " + title)
        
        # Extract title (remove CVE part from title)
        vuln_title = re.sub(r'\s*\(CVE-\d{4}-\d+\)\s*$', '', title).strip()
        if not vuln_title:
            vuln_title = title.strip() if title else None
        
        # Extract description from summary
        description = None
        if summary:
            soup = BeautifulSoup(summary, 'html.parser')
            desc_text = soup.get_text(separator=' ', strip=True)
            # Remove CVSS score part if present
            description = re.sub(r'\s*CVSS[:\s]*\d+\.?\d*\s*$', '', desc_text).strip()
            if not description:
                description = desc_text if desc_text else None
        
        return {
            "found": True,
            "vendor": VENDOR_CONFIG["name"],
            "cve_id": cve_id,
            "title": vuln_title,
            "description": description,
            "source": "rss",
            "advisory_url": entry.link,
            "publish_date": entry.published if hasattr(entry, 'published') else None,
            "severity": severity,
            "score": score,
            "impacted": score >= 7 if score is not None else None,
            "_parsed_date": pub_date
        }
    
    # Step 1: Search titles first
    title_matches = []
    for entry in feed.entries:
        title = entry.title if hasattr(entry, 'title') else ""
        if cve_id.lower() in title.lower():
            title_matches.append(build_result(entry, "title"))
    
    if title_matches:
        # Sort by date (most recent first), None dates go last
        title_matches.sort(key=lambda x: x["_parsed_date"] or datetime.min, reverse=True)
        result = title_matches[0]
        del result["_parsed_date"]
        return result
    
    # Step 2: Search summaries if not found in titles
    summary_matches = []
    for entry in feed.entries:
        summary = entry.summary if hasattr(entry, 'summary') else ""
        if cve_id.lower() in summary.lower():
            summary_matches.append(build_result(entry, "summary"))
    
    if summary_matches:
        # Sort by date (most recent first), None dates go last
        summary_matches.sort(key=lambda x: x["_parsed_date"] or datetime.min, reverse=True)
        result = summary_matches[0]
        del result["_parsed_date"]
        return result

    return {"found": False, "vendor": VENDOR_CONFIG["name"], "cve_id": cve_id, "impacted": None}


def extract_cvss(text):
    """Extract CVSS score and severity from text. Only matches actual CVSS scores, not version numbers."""
    score = None
    severity = None
    
    # Look for CVSS score patterns (must be near CVSS keyword or in score context)
    cvss_patterns = [
        r'CVSS[:\s]*v?\d*\.?\d*[:\s]*(\d+\.?\d*)',  # "CVSS 9.0", "CVSS: 9.0", "CVSSv3: 9.0"
        r'(?:score|rating)[:\s]*(\d+\.?\d*)',        # "score: 9.0", "rating: 9.0"
        r'(\d+\.\d+)\s*/\s*10',                      # "9.0/10"
        r'(\d+\.\d+)\s*\((?:Critical|High|Medium|Low)\)',  # "9.0 (Critical)"
    ]
    
    for pattern in cvss_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0 <= val <= 10:
                    score = val
                    break
            except:
                pass
    
    # Extract severity
    severity_match = re.search(r'\b(Critical|High|Medium|Low)\b', text, re.I)
    if severity_match:
        severity = severity_match.group(1).capitalize()

    return severity, score


def deep_scan(cve_id):
    """Scan security bulletin pages for CVE with expanded content handling."""
    base_url = VENDOR_CONFIG["base_url"]
    bulletin_url = VENDOR_CONFIG["bulletin_url"]
    link_pattern = VENDOR_CONFIG["link_pattern"]

    page = requests.get(bulletin_url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(page.text, "html.parser")

    advisories = []

    # Collect all advisory links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if link_pattern in href:
            # Skip feed/rss URLs
            if "/feed" in href.lower() or "/rss" in href.lower():
                continue
            if href.startswith("http"):
                advisories.append(href)
            else:
                advisories.append(base_url + href)

    # Remove duplicates and sort for consistent ordering
    advisories = sorted(list(set(advisories)))
    
    # Collect all matches and return the best one
    matches = []

    for advisory in advisories:
        result = inspect_advisory(advisory, cve_id)
        if result["found"]:
            matches.append(result)
    
    if matches:
        # Sort by score (highest first) then by severity
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, None: 0}
        matches.sort(
            key=lambda x: (
                x.get("score") or 0,
                severity_order.get((x.get("severity") or "").lower(), 0)
            ),
            reverse=True
        )
        return matches[0]

    return {"found": False, "vendor": VENDOR_CONFIG["name"], "cve_id": cve_id, "impacted": None}


def inspect_advisory(url, cve_id):
    """Inspect advisory page for CVE information - extracts CVSS scores and severity."""
    vendor_name = VENDOR_CONFIG["name"]
    try:
        page = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(page.text, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        
        # Get the full page text
        full_text = soup.get_text(separator=" ", strip=True)

        if cve_id.lower() not in full_text.lower():
            return {"found": False, "vendor": vendor_name, "cve_id": cve_id, "impacted": None}

        # Find all positions where this CVE is mentioned
        cve_positions = [m.start() for m in re.finditer(re.escape(cve_id), full_text, re.IGNORECASE)]
        
        context_scores = []
        context_severities = []
        
        # For each CVE mention, search for CVSS scores and severities nearby
        for pos in cve_positions:
            # Look at text within 500 chars around the CVE mention
            start = max(0, pos - 150)
            end = min(len(full_text), pos + 500)
            context = full_text[start:end]
            
            # Find all CVSS scores in this context
            score_patterns = [
                r'CVSS[:\s]*v?\d*\.?\d*[:\s]*(\d+\.?\d*)',
                r'(?:score|rating)[:\s]*(\d+\.?\d*)',
                r'(\d+\.\d+)\s*/\s*10',
                r'(\d+\.\d+)\s*\((?:Critical|High|Medium|Low)\)',
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                for s in matches:
                    try:
                        val = float(s)
                        if 0 <= val <= 10:
                            context_scores.append(val)
                    except:
                        pass
            
            # Find severities in context
            sevs = re.findall(r'\b(Critical|High|Medium|Low)\b', context, re.IGNORECASE)
            context_severities.extend([s.capitalize() for s in sevs])
        
        # Take the highest score found near the CVE (most severe/latest assessment)
        score = max(context_scores) if context_scores else None
        
        # Take the highest severity found
        severity = None
        if context_severities:
            severity_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
            context_severities.sort(key=lambda x: severity_order.get(x, 0), reverse=True)
            severity = context_severities[0]
        
        # Fallback: if no score found near CVE, search entire page
        if not score:
            all_scores = re.findall(r'CVSS[:\s]*v?\d*\.?\d*[:\s]*(\d+\.?\d*)', full_text, re.IGNORECASE)
            valid_scores = []
            for s in all_scores:
                try:
                    val = float(s)
                    if 0 <= val <= 10:
                        valid_scores.append(val)
                except:
                    pass
            if valid_scores:
                score = max(valid_scores)
        
        # Extract title from page
        page_title = soup.find('title')
        vuln_title = None
        if page_title:
            vuln_title = page_title.get_text(strip=True)
            # Clean up common suffixes
            vuln_title = re.sub(r'\s*[-|]\s*Dynatrace.*$', '', vuln_title).strip()
        
        # Extract description - look for first substantial paragraph or meta description
        description = None
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc['content'].strip()
        if not description:
            # Try to find first paragraph with substantial content
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50 and cve_id.lower() in text.lower():
                    description = text
                    break

        return {
            "found": True,
            "vendor": vendor_name,
            "cve_id": cve_id,
            "title": vuln_title,
            "description": description,
            "source": "security-bulletin",
            "advisory_url": url,
            "severity": severity,
            "score": score,
            "impacted": score >= 7 if score is not None else None
        }
    except Exception as e:
        return {"found": False, "vendor": vendor_name, "cve_id": cve_id, "impacted": None, "error": str(e)}


def analyze(cve_id):
    result = check_rss(cve_id)
    if result["found"]:
        return result

    return deep_scan(cve_id)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python dynatrace_cve_analysis.py <CVE-ID>")
        sys.exit(1)

    cve_id = sys.argv[1]
    result = analyze(cve_id)

    print(json.dumps(result, indent=2))

    if result.get("score", 0) and result["score"] >= 7:
        sys.exit(2)
