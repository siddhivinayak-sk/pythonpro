"""
DefectDojo API Client for retrieving vulnerability findings.

This module provides a clean interface to query DefectDojo findings
and format them for LLM context generation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import requests
import os


@dataclass
class Finding:
    """Model representing a DefectDojo finding."""
    name: str
    severity: str
    cwe: Optional[int]
    vulnerability_id: Optional[str]
    epss: Optional[float]
    date: Optional[str]
    age: Optional[int]
    found_by: str
    status: str
    product: str
    reporter: Optional[str]


class DefectDojoClient:
    """Client for interacting with DefectDojo API."""

    def __init__(self, host: str, api_key: str):
        """
        Initialize DefectDojo client.

        Args:
            host: DefectDojo instance URL
            api_key: API key for authentication
        """
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def get_findings(
        self,
        vulnerability_id: Optional[str] = None,
        component_name: Optional[str] = None,
        component_version: Optional[str] = None,
        cwe: Optional[int] = None,
        severity: Optional[str] = None,
        product: Optional[int] = None,
        product_type: Optional[int] = None,
        status: Optional[str] = None,
        active: Optional[bool] = None,
        verified: Optional[bool] = None,
        epss: Optional[float] = None,
        limit: int = 100
    ) -> list[Finding]:
        """
        Retrieve findings from DefectDojo based on optional filters.

        Args:
            vulnerability_id: CVE or vulnerability identifier
            component_name: Name of the affected component
            component_version: Version of the affected component
            cwe: Common Weakness Enumeration ID
            severity: Severity level (Critical, High, Medium, Low, Info)
            product: Product ID in DefectDojo
            product_type: Product type ID in DefectDojo
            status: Finding status
            active: Filter by active status
            verified: Filter by verified status
            epss: EPSS score filter
            limit: Maximum number of results to return

        Returns:
            List of Finding objects matching the criteria
        """
        params = {
            "limit": limit,
            "prefetch": ["reporter", "test"]
        }

        # Build query parameters only for provided values
        param_mapping = {
            "vulnerability_id": vulnerability_id,
            "component_name": component_name,
            "component_version": component_version,
            "cwe": cwe,
            "severity": severity,
            "product": product,
            "product_type": product_type,
            "status": status,
            "active": active,
            "verified": verified,
            "epss_score": epss,
        }

        for key, value in param_mapping.items():
            if value is not None:
                params[key] = value

        url = f"{self.host}/api/v2/findings/"
        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        prefetch = data.get("prefetch", {})

        # Build lookup maps from prefetch data
        reporter_map = self._build_reporter_map(prefetch.get("reporter", {}))
        test_map = prefetch.get("test", {})

        # Collect unique engagement IDs from tests
        engagement_ids = set()
        for test_id, test_data in test_map.items():
            if isinstance(test_data, dict) and test_data.get("engagement"):
                engagement_ids.add(test_data["engagement"])

        # Fetch engagement details to get product info
        engagement_map = self._fetch_engagements(engagement_ids)

        # Collect unique product IDs from engagements
        product_ids = set()
        for eng_data in engagement_map.values():
            if eng_data.get("product"):
                product_ids.add(eng_data["product"])

        # Fetch product details
        product_map = self._fetch_products(product_ids)

        return self._parse_findings(results, reporter_map, test_map, engagement_map, product_map)

    def _build_reporter_map(self, reporter_data: dict) -> dict[int, str]:
        """Build a mapping of reporter ID to reporter name."""
        reporter_map = {}
        for reporter_id, reporter_info in reporter_data.items():
            if isinstance(reporter_info, dict):
                name_parts = [
                    reporter_info.get("first_name", ""),
                    reporter_info.get("last_name", "")
                ]
                full_name = " ".join(p for p in name_parts if p).strip()
                if not full_name:
                    full_name = reporter_info.get("username", "")
                reporter_map[int(reporter_id)] = full_name
        return reporter_map

    def _fetch_engagements(self, engagement_ids: set[int]) -> dict[int, dict]:
        """Fetch engagement details for given IDs."""
        engagement_map = {}
        if not engagement_ids:
            return engagement_map

        for eng_id in engagement_ids:
            try:
                url = f"{self.host}/api/v2/engagements/{eng_id}/"
                response = self.session.get(url)
                if response.status_code == 200:
                    engagement_map[eng_id] = response.json()
            except requests.RequestException:
                continue
        return engagement_map

    def _fetch_products(self, product_ids: set[int]) -> dict[int, dict]:
        """Fetch product details for given IDs."""
        product_map = {}
        if not product_ids:
            return product_map

        for prod_id in product_ids:
            try:
                url = f"{self.host}/api/v2/products/{prod_id}/"
                response = self.session.get(url)
                if response.status_code == 200:
                    product_map[prod_id] = response.json()
            except requests.RequestException:
                continue
        return product_map

    def _parse_findings(
        self,
        results: list[dict],
        reporter_map: dict[int, str],
        test_map: dict,
        engagement_map: dict[int, dict],
        product_map: dict[int, dict]
    ) -> list[Finding]:
        """Parse API response into Finding objects with enriched data."""
        findings = []
        for item in results:
            # Get reporter name from prefetch
            reporter_id = item.get("reporter")
            reporter_name = reporter_map.get(reporter_id) if reporter_id else None

            # Get product name by traversing test -> engagement -> product
            product_name = self._resolve_product_name(
                item.get("test"),
                test_map,
                engagement_map,
                product_map
            )

            finding = Finding(
                name=item.get("title", ""),
                severity=item.get("severity", ""),
                cwe=item.get("cwe"),
                vulnerability_id=self._extract_vulnerability_id(item),
                epss=item.get("epss_score"),
                date=item.get("date"),
                age=item.get("age"),
                found_by=self._get_found_by(item),
                status=self._determine_status(item),
                product=product_name,
                reporter=reporter_name
            )
            findings.append(finding)
        return findings

    def _resolve_product_name(
        self,
        test_id: Optional[int],
        test_map: dict,
        engagement_map: dict[int, dict],
        product_map: dict[int, dict]
    ) -> str:
        """Resolve product name from test -> engagement -> product chain."""
        if not test_id:
            return ""

        test_data = test_map.get(str(test_id), {})
        if not isinstance(test_data, dict):
            return ""

        engagement_id = test_data.get("engagement")
        if not engagement_id:
            return ""

        engagement_data = engagement_map.get(engagement_id, {})
        product_id = engagement_data.get("product")
        if not product_id:
            return ""

        product_data = product_map.get(product_id, {})
        return product_data.get("name", "")

    def _extract_vulnerability_id(self, item: dict) -> Optional[str]:
        """Extract CVE or vulnerability ID from finding."""
        vuln_ids = item.get("vulnerability_ids", [])
        if vuln_ids:
            return vuln_ids[0].get("vulnerability_id") if isinstance(vuln_ids[0], dict) else vuln_ids[0]
        return item.get("cve")

    def _get_found_by(self, item: dict) -> str:
        """Get the scanner/tool that found the vulnerability."""
        found_by = item.get("found_by", [])
        if found_by and isinstance(found_by, list):
            return ", ".join(str(f) for f in found_by)
        return str(found_by) if found_by else ""

    def _determine_status(self, item: dict) -> str:
        """Determine finding status from boolean flags."""
        if item.get("is_mitigated"):
            return "Mitigated"
        if item.get("risk_accepted"):
            return "Risk Accepted"
        if item.get("false_p"):
            return "False Positive"
        if item.get("out_of_scope"):
            return "Out of Scope"
        if item.get("verified"):
            return "Verified"
        if item.get("active"):
            return "Active"
        return "Inactive"


def findings_to_markdown(findings: list[Finding], title: str = "Vulnerability Findings") -> str:
    """
    Convert findings to markdown format for LLM context, grouped by product.

    Args:
        findings: List of Finding objects
        title: Title for the markdown document

    Returns:
        Markdown formatted string with findings grouped by product
    """
    if not findings:
        return f"# {title}\n\nNo findings match the specified criteria."

    # Group findings by product
    products: dict[str, list[Finding]] = {}
    for f in findings:
        product_name = f.product or "Unknown Product"
        if product_name not in products:
            products[product_name] = []
        products[product_name].append(f)

    lines = [f"# {title}", "", f"**Total Findings:** {len(findings)}", f"**Impacted Products:** {len(products)}", ""]

    # Summary by severity
    severity_counts = {}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    if severity_counts:
        lines.append("## Summary by Severity")
        for sev in ["Critical", "High", "Medium", "Low", "Info"]:
            if sev in severity_counts:
                lines.append(f"- **{sev}:** {severity_counts[sev]}")
        lines.append("")

    # Impacted products summary
    lines.append("## Impacted Products")
    lines.append("")
    for product_name, product_findings in products.items():
        lines.append(f"- **{product_name}**: {len(product_findings)} component(s) affected")
    lines.append("")

    # Detailed findings grouped by product
    lines.append("## Impacted Components by Product")
    lines.append("")

    for product_name, product_findings in products.items():
        lines.append(f"### {product_name}")
        lines.append("")
        lines.append("| # | Title | Severity | Vuln ID | CWE | EPSS | Status | Date | Age | Found By | Reporter |")
        lines.append("|---|-------|----------|---------|-----|------|--------|------|-----|----------|----------|")

        for i, f in enumerate(product_findings, 1):
            title_text = f.name[:50] + "..." if len(f.name) > 50 else f.name
            vuln_id = f.vulnerability_id or "N/A"
            cwe = f"CWE-{f.cwe}" if f.cwe else "N/A"
            epss = f"{f.epss:.4f}" if f.epss is not None else "N/A"
            age = str(f.age) if f.age is not None else "N/A"
            found_by = f.found_by or "N/A"
            reporter = f.reporter or "N/A"
            lines.append(f"| {i} | {title_text} | {f.severity} | {vuln_id} | {cwe} | {epss} | {f.status} | {f.date or 'N/A'} | {age} | {found_by} | {reporter} |")

        lines.append("")

    return "\n".join(lines)


def findings_to_compact_markdown(findings: list[Finding], title: str = "Vulnerability Findings") -> str:
    """
    Convert findings to a compact markdown table format for LLM context.
    Lists all impacted components as rows in a single table.

    Args:
        findings: List of Finding objects
        title: Title for the markdown document

    Returns:
        Compact markdown formatted string with single table of impacted components
    """
    if not findings:
        return f"# {title}\n\nNo findings match the specified criteria."

    lines = [
        f"# {title}",
        "",
        f"**Total:** {len(findings)}",
        "",
        "| # | Title | Severity | Vuln ID | CWE | Product | Status | Date |",
        "|---|-------|----------|---------|-----|---------|--------|------|"
    ]

    for i, f in enumerate(findings, 1):
        title_text = f.name[:40] + "..." if len(f.name) > 40 else f.name
        vuln_id = f.vulnerability_id or "N/A"
        cwe = f"CWE-{f.cwe}" if f.cwe else "N/A"
        lines.append(f"| {i} | {title_text} | {f.severity} | {vuln_id} | {cwe} | {f.product} | {f.status} | {f.date or 'N/A'} |")

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    host = os.getenv('MY_DEFECT_DOJO_HOST')
    key = os.getenv('MY_DEFECT_DOJO_KEY')

    # Initialize client
    client = DefectDojoClient(
        host=host,
        api_key=key
    )

    # Search for findings by CVE
    findings = client.get_findings(
        vulnerability_id="CVE-2016-1000027",
        severity="Critical",
        active=True
    )

    # Generate markdown context for LLM
    markdown_context = findings_to_markdown(findings, title="Critical CVE-2016-1000027 Findings")
    print(markdown_context)

    # Or use compact format
    compact_context = findings_to_compact_markdown(findings)
    print(compact_context)
