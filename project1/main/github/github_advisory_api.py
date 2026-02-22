"""
GitHub Global Security Advisory API Client

This module provides functionality to consume GitHub's Global Security Advisory API
and transform the results into structured models suitable for LLM context.
"""

import os
import requests
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from enum import Enum


class AdvisoryType(str, Enum):
    """Advisory type enumeration."""
    REVIEWED = "reviewed"
    MALWARE = "malware"
    UNREVIEWED = "unreviewed"


class Severity(str, Enum):
    """Severity level enumeration."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class Vulnerability:
    """Represents a vulnerability within an advisory."""
    package: Optional[str] = None
    version_range: Optional[str] = None
    first_patch_version: Optional[str] = None
    vulnerable_functions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CvssSeverity:
    """CVSS severity scores."""
    cvss_v3: Optional[str] = None
    cvss_v4: Optional[str] = None
    score_v3: Optional[float] = None
    score_v4: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Cwe:
    """Common Weakness Enumeration."""
    cwe_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Epss:
    """Exploit Prediction Scoring System."""
    percentage: Optional[float] = None
    percentile: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SecurityAdvisory:
    """
    Security Advisory model optimized for LLM context.
    
    Contains essential CVE/GHSA information for vulnerability analysis.
    """
    ghsa_id: Optional[str] = None
    cve_id: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    severity: Optional[str] = None
    repository_url: Optional[str] = None
    source_code_location: Optional[str] = None
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    cvss_severities: Optional[CvssSeverity] = None
    cwes: List[Cwe] = field(default_factory=list)
    cvss: Optional[str] = None
    epss: Optional[Epss] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "ghsa_id": self.ghsa_id,
            "cve_id": self.cve_id,
            "url": self.url,
            "summary": self.summary,
            "description": self.description,
            "type": self.type,
            "severity": self.severity,
            "repository_url": self.repository_url,
            "source_code_location": self.source_code_location,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "cvss_severities": self.cvss_severities.to_dict() if self.cvss_severities else None,
            "cwes": [c.to_dict() for c in self.cwes],
            "cvss": self.cvss,
            "epss": self.epss.to_dict() if self.epss else None,
        }
        return result

    def to_llm_context(self) -> str:
        """
        Format advisory as concise text for LLM context.
        
        Returns:
            Formatted string with key vulnerability details.
        """
        lines = [f"Security Advisory: {self.ghsa_id or self.cve_id}"]
        
        if self.cve_id:
            lines.append(f"CVE: {self.cve_id}")
        if self.severity:
            lines.append(f"Severity: {self.severity.upper()}")
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.vulnerabilities:
            lines.append("Affected Packages:")
            for vuln in self.vulnerabilities:
                pkg_info = f"  - {vuln.package}"
                if vuln.version_range:
                    pkg_info += f" ({vuln.version_range})"
                if vuln.first_patch_version:
                    pkg_info += f" | Fixed in: {vuln.first_patch_version}"
                lines.append(pkg_info)
        if self.cwes:
            cwe_list = ", ".join([c.cwe_id for c in self.cwes if c.cwe_id])
            if cwe_list:
                lines.append(f"CWEs: {cwe_list}")
        if self.url:
            lines.append(f"Reference: {self.url}")
        
        return "\n".join(lines)


class GitHubAdvisoryClient:
    """
    Client for GitHub Global Security Advisory API.
    
    Fetches and transforms security advisories into structured models.
    """
    
    BASE_URL = "https://api.github.com/advisories"
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            token: GitHub personal access token. If not provided,
                   reads from GITHUB_TOKEN environment variable.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        self._setup_headers()

    def _setup_headers(self) -> None:
        """Configure request headers."""
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _parse_vulnerability(self, vuln_data: dict) -> Vulnerability:
        """Parse vulnerability data from API response."""
        package_data = vuln_data.get("package", {}) or {}
        return Vulnerability(
            package=package_data.get("name"),
            version_range=vuln_data.get("vulnerable_version_range"),
            first_patch_version=vuln_data.get("first_patched_version"),
            vulnerable_functions=vuln_data.get("vulnerable_functions") or [],
        )

    def _parse_cvss_severities(self, data: dict) -> Optional[CvssSeverity]:
        """Parse CVSS severity data."""
        cvss_data = data.get("cvss_severities") or {}
        cvss_v3 = cvss_data.get("cvss_v3") or {}
        cvss_v4 = cvss_data.get("cvss_v4") or {}
        
        if not cvss_v3 and not cvss_v4:
            return None
        
        return CvssSeverity(
            cvss_v3=cvss_v3.get("vector_string"),
            cvss_v4=cvss_v4.get("vector_string"),
            score_v3=cvss_v3.get("score"),
            score_v4=cvss_v4.get("score"),
        )

    def _parse_cwes(self, data: dict) -> List[Cwe]:
        """Parse CWE data from API response."""
        cwes_data = data.get("cwes") or []
        return [
            Cwe(cwe_id=cwe.get("cwe_id"), name=cwe.get("name"))
            for cwe in cwes_data
        ]

    def _parse_epss(self, data: dict) -> Optional[Epss]:
        """Parse EPSS data from API response."""
        epss_data = data.get("epss")
        if not epss_data:
            return None
        return Epss(
            percentage=epss_data.get("percentage"),
            percentile=epss_data.get("percentile"),
        )

    def _parse_advisory(self, data: dict) -> SecurityAdvisory:
        """
        Parse API response into SecurityAdvisory model.
        
        Args:
            data: Raw API response dictionary.
            
        Returns:
            Populated SecurityAdvisory instance.
        """
        vulnerabilities = [
            self._parse_vulnerability(v) 
            for v in (data.get("vulnerabilities") or [])
        ]
        
        # Extract source code location from identifiers or references
        source_code_location = None
        for ref in data.get("references") or []:
            if ref and "github.com" in ref and "/tree/" in ref:
                source_code_location = ref
                break
        
        return SecurityAdvisory(
            ghsa_id=data.get("ghsa_id"),
            cve_id=data.get("cve_id"),
            url=data.get("html_url"),
            summary=data.get("summary"),
            description=data.get("description"),
            type=data.get("type"),
            severity=data.get("severity"),
            repository_url=data.get("source_code_location") or data.get("repository_advisory_url"),
            source_code_location=source_code_location,
            published_at=data.get("published_at"),
            updated_at=data.get("updated_at"),
            vulnerabilities=vulnerabilities,
            cvss_severities=self._parse_cvss_severities(data),
            cwes=self._parse_cwes(data),
            cvss=data.get("cvss", {}).get("vector_string") if data.get("cvss") else None,
            epss=self._parse_epss(data),
        )

    def get_advisory(
        self,
        ghsa_id: Optional[str] = None,
        cve_id: Optional[str] = None,
    ) -> Optional[SecurityAdvisory]:
        """
        Fetch a single advisory by GHSA ID or CVE ID.
        
        Args:
            ghsa_id: GitHub Security Advisory ID (e.g., "GHSA-xxxx-xxxx-xxxx").
            cve_id: CVE identifier (e.g., "CVE-2024-1234").
            
        Returns:
            SecurityAdvisory if found, None otherwise.
            
        Raises:
            ValueError: If neither ghsa_id nor cve_id is provided.
            requests.HTTPError: On API errors.
        """
        if not ghsa_id and not cve_id:
            raise ValueError("Either ghsa_id or cve_id must be provided")
        
        advisory_id = ghsa_id or cve_id
        url = f"{self.BASE_URL}/{advisory_id}"
        
        response = self.session.get(url)
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        return self._parse_advisory(response.json())

    def search_advisories(
        self,
        ghsa_id: Optional[str] = None,
        cve_id: Optional[str] = None,
        type: Optional[str] = None,
        severity: Optional[str] = None,
        ecosystem: Optional[str] = None,
        package: Optional[str] = None,
        affects: Optional[str] = None,
        cwes: Optional[List[str]] = None,
        is_withdrawn: Optional[bool] = None,
        published: Optional[str] = None,
        updated: Optional[str] = None,
        modified: Optional[str] = None,
        per_page: int = 30,
        cursor: Optional[str] = None,
    ) -> List[SecurityAdvisory]:
        """
        Search advisories with optional filters.
        
        All parameters are optional. Only provided parameters are included in the query.
        
        Args:
            ghsa_id: Filter by GHSA ID.
            cve_id: Filter by CVE ID.
            type: Filter by type ("reviewed", "malware", "unreviewed").
            severity: Filter by severity ("critical", "high", "medium", "low", "unknown").
            ecosystem: Filter by ecosystem (e.g., "npm", "pip", "maven").
            package: Filter by package name.
            affects: Filter by affected version string (e.g., "package@version").
            cwes: Filter by list of CWE IDs.
            is_withdrawn: Filter by withdrawn status.
            published: Filter by publish date (ISO 8601 format or range).
            updated: Filter by update date (ISO 8601 format or range).
            modified: Filter by modification date (ISO 8601 format or range).
            per_page: Number of results per page (max 100).
            cursor: Pagination cursor for fetching next page.
            
        Returns:
            List of SecurityAdvisory objects matching the criteria.
            
        Raises:
            requests.HTTPError: On API errors.
        """
        params = {}
        
        # Build query params only for provided values
        if ghsa_id:
            params["ghsa_id"] = ghsa_id
        if cve_id:
            params["cve_id"] = cve_id
        if type:
            params["type"] = type
        if severity:
            params["severity"] = severity
        if ecosystem:
            params["ecosystem"] = ecosystem
        if package:
            params["package"] = package
        if affects:
            params["affects"] = affects
        if cwes:
            params["cwes"] = ",".join(cwes)
        if is_withdrawn is not None:
            params["is_withdrawn"] = str(is_withdrawn).lower()
        if published:
            params["published"] = published
        if updated:
            params["updated"] = updated
        if modified:
            params["modified"] = modified
        if per_page:
            params["per_page"] = min(per_page, 100)
        if cursor:
            params["after"] = cursor
        
        response = self.session.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        return [self._parse_advisory(item) for item in response.json()]

    def get_advisories_for_llm_context(
        self,
        ghsa_id: Optional[str] = None,
        cve_id: Optional[str] = None,
        type: Optional[str] = None,
        severity: Optional[str] = None,
        max_results: int = 10,
    ) -> str:
        """
        Fetch advisories and format them as LLM context.
        
        Convenience method that searches advisories and returns
        formatted text suitable for LLM prompts.
        
        Args:
            ghsa_id: Filter by GHSA ID.
            cve_id: Filter by CVE ID.
            type: Filter by advisory type.
            severity: Filter by severity level.
            max_results: Maximum number of advisories to include.
            
        Returns:
            Formatted string containing advisory details.
        """
        advisories = self.search_advisories(
            ghsa_id=ghsa_id,
            cve_id=cve_id,
            type=type,
            severity=severity,
            per_page=max_results,
        )
        
        if not advisories:
            return "No security advisories found matching the criteria."
        
        context_parts = [
            f"Found {len(advisories)} security advisory(ies):\n",
            "-" * 50,
        ]
        
        for advisory in advisories:
            context_parts.append(advisory.to_llm_context())
            context_parts.append("-" * 50)
        
        return "\n".join(context_parts)


# Convenience functions for direct usage
def fetch_advisory(
    ghsa_id: Optional[str] = None,
    cve_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[SecurityAdvisory]:
    """
    Fetch a single security advisory.
    
    Args:
        ghsa_id: GitHub Security Advisory ID.
        cve_id: CVE identifier.
        token: GitHub API token (optional).
        
    Returns:
        SecurityAdvisory if found, None otherwise.
    """
    client = GitHubAdvisoryClient(token=token)
    return client.get_advisory(ghsa_id=ghsa_id, cve_id=cve_id)


def search_advisories(
    ghsa_id: Optional[str] = None,
    cve_id: Optional[str] = None,
    type: Optional[str] = None,
    severity: Optional[str] = None,
    ecosystem: Optional[str] = None,
    token: Optional[str] = None,
) -> List[SecurityAdvisory]:
    """
    Search security advisories with optional filters.
    
    Args:
        ghsa_id: Filter by GHSA ID.
        cve_id: Filter by CVE ID.
        type: Filter by advisory type.
        severity: Filter by severity level.
        ecosystem: Filter by ecosystem.
        token: GitHub API token (optional).
        
    Returns:
        List of matching SecurityAdvisory objects.
    """
    client = GitHubAdvisoryClient(token=token)
    return client.search_advisories(
        ghsa_id=ghsa_id,
        cve_id=cve_id,
        type=type,
        severity=severity,
        ecosystem=ecosystem,
    )


if __name__ == "__main__":
    # Example usage
    client = GitHubAdvisoryClient()
    
    # 1. Search for critical vulnerabilities
    # advisories = client.search_advisories(severity="critical", per_page=5)
    # for advisory in advisories:
    #     print(advisory.to_llm_context())
    #     print("-" * 50)


    # 2. Search by CVE ID
    advisories = client.search_advisories(cve_id="CVE-2026-27198", per_page=5)
    for advisory in advisories:
        print(advisory.to_llm_context())
        print("-" * 50)

