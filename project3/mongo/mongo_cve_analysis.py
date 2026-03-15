import requests
import feedparser
import re
from bs4 import BeautifulSoup
import sys
import json
from datetime import datetime
from dateutil import parser as date_parser


# ===============================
# MongoDB Configuration
# ===============================
VENDOR_CONFIG = {
    "name": "MongoDB",
    "base_url": "https://www.mongodb.com",
    "rss_url": "https://www.mongodb.com/alerts/rss",
    "bulletin_url": "https://www.mongodb.com/resources/products/alerts#security",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def check_rss(cve_id):
    """
    Check RSS feed for CVE.
    RSS structure:
    <item>
      <title>... (CVE-XXXX-XXXXX)</title>
      <description><![CDATA[...<br> CVSS Score has been rated as X.X ]]></description>
      <pubDate>...</pubDate>
      <link>...</link>
    </item>
    """
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
            
            # Extract CVSS score from description (after <br> tag)
            # Pattern: "CVSS Score has been rated as X.X"
            score = None
            score_match = re.search(r'CVSS\s+Score\s+has\s+been\s+rated\s+as\s+(\d+\.?\d*)', summary, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except:
                    pass
            
            # Extract title (remove CVE part from title)
            vuln_title = re.sub(r'\s*\(CVE-\d{4}-\d+\)\s*$', '', title).strip()
            
            # Extract description (remove CVSS score part from summary)
            # Parse HTML from summary to get clean text
            soup = BeautifulSoup(summary, 'html.parser')
            desc_text = soup.get_text(separator=' ', strip=True)
            # Remove CVSS score part
            description = re.sub(r'\s*CVSS\s+Score\s+has\s+been\s+rated\s+as\s+\d+\.?\d*\s*$', '', desc_text).strip()
            
            # Try to extract affected product from description
            affected_product = None
            product_match = re.search(r'MongoDB\s+([A-Za-z#\+]+(?:\s+[A-Za-z]+)?)\s+(?:Driver|Server|Client|Shell)', desc_text, re.IGNORECASE)
            if product_match:
                affected_product = f"MongoDB {product_match.group(1)} Driver" if 'Driver' in desc_text else product_match.group(0)
            else:
                # Look for "This issue affects:" pattern
                product_match = re.search(r'affects[:\s]+([^\.]+)', desc_text, re.IGNORECASE)
                if product_match:
                    affected_product = product_match.group(1).strip()
            
            matches.append({
                "found": True,
                "vendor": VENDOR_CONFIG["name"],
                "cve_id": cve_id,
                "title": vuln_title,
                "description": description,
                "affected_product": affected_product,
                "affected_versions": None,  # Not reliably available in RSS
                "source": "rss",
                "advisory_url": entry.link,
                "publish_date": entry.published if hasattr(entry, 'published') else None,
                "score": score,
                "impacted": score >= 7 if score is not None else None,
                "_parsed_date": pub_date
            })
    
    if matches:
        # Sort by date (most recent first), None dates go last
        matches.sort(key=lambda x: x["_parsed_date"] or datetime.min, reverse=True)
        result = matches[0]
        del result["_parsed_date"]  # Remove internal field
        return result

    return {"found": False, "vendor": VENDOR_CONFIG["name"], "cve_id": cve_id, "impacted": None}


def deep_scan(cve_id):
    """
    Scan security bulletin page for CVE.
    Bulletin structure - each CVE is in a card with:
    - Date in div with class containing "Eyebrow"
    - CVE in div with class containing "Eyebrow"  
    - Score in div with class containing "ScoreBox"
    - Title in h4 tag
    - Description in ContentDiv
    - Affected product/versions in ProductWithVersionsDiv
    - Reference link in footer
    """
    bulletin_url = VENDOR_CONFIG["bulletin_url"]

    page = requests.get(bulletin_url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(page.text, "html.parser")

    # Find all card containers
    cards = soup.find_all("div", class_=re.compile(r'AlertsCardGrid__CardContainer'))
    
    for card in cards:
        # Find CVE in eyebrow divs
        eyebrows = card.find_all("div", class_=re.compile(r'AlertsCardGrid__Eyebrow'))
        card_cve = None
        card_date = None
        
        for eyebrow in eyebrows:
            text = eyebrow.get_text(strip=True)
            # Check if this eyebrow contains CVE
            cve_match = re.search(r'(CVE-\d{4}-\d+)', text, re.IGNORECASE)
            if cve_match:
                card_cve = cve_match.group(1).upper()
            # Check if this eyebrow contains date (MM/DD/YYYY format)
            elif re.match(r'\d{2}/\d{2}/\d{4}', text):
                card_date = text
        
        # Check if this card matches our CVE
        if card_cve and card_cve.upper() == cve_id.upper():
            # Extract score from ScoreBox
            score = None
            score_box = card.find("div", class_=re.compile(r'AlertsCardGrid__ScoreBox'))
            if score_box:
                score_text = score_box.get_text(strip=True)
                try:
                    score = float(score_text)
                except:
                    pass
            
            # Extract title from h4
            vuln_title = None
            h4 = card.find("h4")
            if h4:
                vuln_title = h4.get_text(strip=True)
            
            # Extract description from ContentDiv
            description = None
            content_div = card.find("div", class_=re.compile(r'AlertsCardGrid__ContentDiv'))
            if content_div:
                # Get first p or span with description text
                desc_p = content_div.find("p")
                if desc_p:
                    description = desc_p.get_text(strip=True)
            
            # Extract affected product and versions from ProductWithVersionsDiv
            affected_product = None
            affected_versions = None
            product_div = card.find("div", class_=re.compile(r'AlertsCardGrid__ProductWithVersionsDiv'))
            if product_div:
                # Look for "Affects:" followed by product name
                all_eyebrows = product_div.find_all("div", class_=re.compile(r'AlertsCardGrid__Eyebrow'))
                for i, eyebrow in enumerate(all_eyebrows):
                    label = eyebrow.get_text(strip=True)
                    # Get the next sibling p tag for the value
                    next_p = eyebrow.find_next_sibling("p")
                    if next_p:
                        value = next_p.get_text(strip=True)
                        if "Affects" in label:
                            affected_product = value
                        elif "Versions" in label:
                            affected_versions = value
            
            # Extract reference link
            advisory_url = None
            footer = card.find("div", class_=re.compile(r'AlertsCardGrid__Footer'))
            if footer:
                link = footer.find("a", href=True)
                if link:
                    advisory_url = link["href"]
            
            return {
                "found": True,
                "vendor": VENDOR_CONFIG["name"],
                "cve_id": cve_id,
                "title": vuln_title,
                "description": description,
                "affected_product": affected_product,
                "affected_versions": affected_versions,
                "source": "security-bulletin",
                "advisory_url": advisory_url,
                "publish_date": card_date,
                "score": score,
                "impacted": score >= 7 if score is not None else None
            }

    return {"found": False, "vendor": VENDOR_CONFIG["name"], "cve_id": cve_id, "impacted": None}


def analyze(cve_id):
    result = check_rss(cve_id)
    if result["found"]:
        return result

    return deep_scan(cve_id)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python mongo_cve_analysis.py <CVE-ID>")
        sys.exit(1)

    cve_id = sys.argv[1]
    result = analyze(cve_id)

    print(json.dumps(result, indent=2))

    if result.get("score", 0) and result["score"] >= 7:
        sys.exit(2)
