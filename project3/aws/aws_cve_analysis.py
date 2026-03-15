import requests
import feedparser
import re
from bs4 import BeautifulSoup
import sys
import json
from datetime import datetime
from dateutil import parser as date_parser


# ===============================
# AWS Configuration
# ===============================
VENDOR_CONFIG = {
    "name": "AWS",
    "base_url": "https://aws.amazon.com",
    "alas_url": "https://explore.alas.aws.amazon.com",
    "rss_url": "https://aws.amazon.com/security/security-bulletins/rss/feed/",
    "bulletin_url": "https://aws.amazon.com/security/security-bulletins/",
    "link_pattern": "/security/security-bulletins/",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def check_alas(cve_id):
    """
    Check AWS ALAS explore page for CVE.
    URL format: https://explore.alas.aws.amazon.com/{CVE-ID}.html
    
    Page structure:
    - Blue header: CVE ID, Public date, Description, Severity, CVSS Score
    - Affected Packages table: Platform, Package, Release Date, Advisory, Status
    - CVSS Scores table: Score Type, Score, Vector
    """
    url = f"{VENDOR_CONFIG['alas_url']}/{cve_id}.html"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            return {"found": False, "vendor": VENDOR_CONFIG["name"], "cve_id": cve_id, "impacted": None}
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Verify page contains the CVE
        title = soup.find("title")
        if not title or cve_id.upper() not in title.text.upper():
            return {"found": False, "vendor": VENDOR_CONFIG["name"], "cve_id": cve_id, "impacted": None}
        
        description = None
        severity = None
        score = None
        public_date = None
        
        # Extract description from .description-content
        desc_elem = soup.find(class_="description-content")
        if desc_elem:
            description = desc_elem.get_text(strip=True)
        
        # Extract public date from .date-display
        date_elem = soup.find(class_="date-display")
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_text)
            if date_match:
                public_date = date_match.group(1)
        
        # Extract severity and score from description-row
        desc_row = soup.find(class_="description-row")
        if desc_row:
            headers = desc_row.find_all(class_="description-header")
            for header in headers:
                header_text = header.get_text(strip=True)
                
                if "Severity" in header_text:
                    # Severity is in severity-icon-container sibling
                    sev_container = header.find_next_sibling(class_="severity-icon-container")
                    if sev_container:
                        sev_div = sev_container.find("div")
                        if sev_div:
                            sev_text = sev_div.get_text(strip=True)
                            if sev_text in ["Critical", "High", "Medium", "Low"]:
                                severity = sev_text
                
                elif "CVSS" in header_text and "Score" in header_text:
                    # Score is in score-icon div
                    score_container = header.find_next_sibling(class_="severity-icon-container")
                    if score_container:
                        score_icon = score_container.find(class_="score-icon")
                        if score_icon:
                            score_text = score_icon.get_text(strip=True)
                            try:
                                score = float(score_text)
                            except:
                                pass
        
        # Extract affected packages from table
        affected_packages = []
        tables = soup.find_all("table")
        if tables:
            # First table is affected packages
            pkg_table = tables[0]
            rows = pkg_table.find_all("tr")[1:]  # Skip header row
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 5:
                    # Normalize whitespace in platform (remove newlines, collapse spaces)
                    platform = ' '.join(cells[0].get_text().split())
                    pkg_info = {
                        "platform": platform,
                        "package": cells[1].get_text(strip=True),
                        "release_date": cells[2].get_text(strip=True),
                        "advisory": cells[3].get_text(strip=True),
                        "status": cells[4].get_text(strip=True),
                    }
                    affected_packages.append(pkg_info)
        
        # Fallback: Extract CVSS from scores table if not found
        if score is None and len(tables) > 1:
            scores_table = tables[1]
            rows = scores_table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    try:
                        score = float(cells[1].get_text(strip=True))
                        break
                    except:
                        pass
        
        return {
            "found": True,
            "vendor": VENDOR_CONFIG["name"],
            "cve_id": cve_id,
            "title": cve_id,
            "description": description,
            "severity": severity,
            "affected_packages": affected_packages if affected_packages else None,
            "source": "alas",
            "advisory_url": url,
            "publish_date": public_date,
            "score": score,
            "impacted": score >= 7 if score is not None else None
        }
        
    except Exception as e:
        return {"found": False, "vendor": VENDOR_CONFIG["name"], "cve_id": cve_id, "impacted": None}


def check_rss(cve_id):
    """Check RSS feed for CVE - returns the most recent matching entry."""
    feed = feedparser.parse(VENDOR_CONFIG["rss_url"])
    
    matches = []
    for entry in feed.entries:
        title = entry.title if hasattr(entry, 'title') else ""
        summary = entry.summary if hasattr(entry, 'summary') else ""
        
        if cve_id.lower() in title.lower() or cve_id.lower() in summary.lower():
            # Parse publish date for sorting
            pub_date = None
            if hasattr(entry, 'published'):
                try:
                    pub_date = date_parser.parse(entry.published)
                except:
                    pass
            
            severity, score = extract_cvss(summary + " " + title)
            impacted = calculate_impacted(severity, score)

            matches.append({
                "found": True,
                "vendor": VENDOR_CONFIG["name"],
                "cve_id": cve_id,
                "source": "rss",
                "advisory_url": entry.link,
                "publish_date": entry.published if hasattr(entry, 'published') else None,
                "severity": severity,
                "score": score,
                "impacted": impacted,
                "_parsed_date": pub_date
            })
    
    if matches:
        # Sort by date (most recent first), None dates go last
        matches.sort(key=lambda x: x["_parsed_date"] or datetime.min, reverse=True)
        result = matches[0]
        del result["_parsed_date"]  # Remove internal field
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


def calculate_impacted(severity, score):
    """
    Determine if CVE is impacted based on severity and score.
    Returns:
        True: if severity is Critical/High OR score >= 7
        False: if CVE found but severity is Low/Medium AND score < 7
        None: if both severity and score are null (cannot determine)
    """
    if severity is None and score is None:
        return None
    
    # Check severity
    severity_impacted = severity and severity.lower() in ["high", "critical"]
    
    # Check score
    score_impacted = score is not None and score >= 7
    
    return severity_impacted or score_impacted


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
        
        impacted = calculate_impacted(severity, score)

        return {
            "found": True,
            "vendor": vendor_name,
            "cve_id": cve_id,
            "source": "security-bulletin",
            "advisory_url": url,
            "severity": severity,
            "score": score,
            "impacted": impacted
        }
    except Exception as e:
        return {"found": False, "vendor": vendor_name, "cve_id": cve_id, "impacted": None, "error": str(e)}


def analyze(cve_id):
    # First try ALAS explore page (most reliable for AWS CVEs)
    result = check_alas(cve_id)
    if result["found"]:
        return result
    
    # Fall back to RSS
    result = check_rss(cve_id)
    if result["found"]:
        return result

    # Fall back to bulletin scan
    return deep_scan(cve_id)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python aws_cve_analysis.py <CVE-ID>")
        sys.exit(1)

    cve_id = sys.argv[1]
    result = analyze(cve_id)

    print(json.dumps(result, indent=2))

    if result.get("score", 0) and result["score"] >= 7:
        sys.exit(2)
