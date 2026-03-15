import requests
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def fetch_nvd_cve(cve_id, api_key=None):
    """Fetch CVE details from NVD API v2.0"""
    
    headers = {}
    if api_key:
        headers["apiKey"] = api_key
    
    params = {"cveId": cve_id}
    
    try:
        response = requests.get(NVD_API_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check if CVE was found
        if data.get("totalResults", 0) == 0:
            return None
        
        # Extract CVE data
        cve_item = data["vulnerabilities"][0]["cve"]
        
        return parse_cve_data(cve_item, cve_id)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch CVE from NVD: {e}")


def parse_cve_data(cve_item, cve_id):
    """Parse and structure CVE data from NVD response"""
    
    result = {
        "cve_id": cve_id,
        "source": "NVD",
        "nvd_url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
        "description": extract_description(cve_item),
        "published_date": cve_item.get("published"),
        "last_modified": cve_item.get("lastModified"),
        "cvss_v3": extract_cvss_v3(cve_item),
        "cvss_v2": extract_cvss_v2(cve_item),
        "cwe": extract_cwe(cve_item),
        "references": extract_references(cve_item),
        "cpe_configurations": extract_cpe_summary(cve_item)
    }
    
    return result


def extract_description(cve_item):
    """Extract English description"""
    descriptions = cve_item.get("descriptions", [])
    for desc in descriptions:
        if desc.get("lang") == "en":
            return desc.get("value", "No description available")
    return "No description available"


def extract_cvss_v3(cve_item):
    """Extract CVSS v3.x metrics"""
    metrics = cve_item.get("metrics", {})
    
    # Try CVSS v3.1 first, then v3.0
    for version in ["cvssMetricV31", "cvssMetricV30"]:
        if version in metrics and len(metrics[version]) > 0:
            cvss_data = metrics[version][0]["cvssData"]
            return {
                "version": cvss_data.get("version"),
                "vector_string": cvss_data.get("vectorString"),
                "base_score": cvss_data.get("baseScore"),
                "base_severity": cvss_data.get("baseSeverity"),
                "attack_vector": cvss_data.get("attackVector"),
                "attack_complexity": cvss_data.get("attackComplexity"),
                "privileges_required": cvss_data.get("privilegesRequired"),
                "user_interaction": cvss_data.get("userInteraction"),
                "scope": cvss_data.get("scope"),
                "confidentiality_impact": cvss_data.get("confidentialityImpact"),
                "integrity_impact": cvss_data.get("integrityImpact"),
                "availability_impact": cvss_data.get("availabilityImpact"),
                "exploitability_score": metrics[version][0].get("exploitabilityScore"),
                "impact_score": metrics[version][0].get("impactScore")
            }
    
    return None


def extract_cvss_v2(cve_item):
    """Extract CVSS v2 metrics if available"""
    metrics = cve_item.get("metrics", {})
    
    if "cvssMetricV2" in metrics and len(metrics["cvssMetricV2"]) > 0:
        cvss_data = metrics["cvssMetricV2"][0]["cvssData"]
        return {
            "version": cvss_data.get("version"),
            "vector_string": cvss_data.get("vectorString"),
            "base_score": cvss_data.get("baseScore"),
            "access_vector": cvss_data.get("accessVector"),
            "access_complexity": cvss_data.get("accessComplexity"),
            "authentication": cvss_data.get("authentication"),
            "confidentiality_impact": cvss_data.get("confidentialityImpact"),
            "integrity_impact": cvss_data.get("integrityImpact"),
            "availability_impact": cvss_data.get("availabilityImpact")
        }
    
    return None


def extract_cwe(cve_item):
    """Extract CWE (Common Weakness Enumeration) classifications"""
    weaknesses = cve_item.get("weaknesses", [])
    cwe_list = []
    
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            if desc.get("lang") == "en":
                cwe_list.append(desc.get("value"))
    
    return cwe_list if cwe_list else ["CWE not specified"]


def extract_references(cve_item):
    """Extract reference URLs (limit to first 10 for token efficiency)"""
    references = cve_item.get("references", [])
    ref_list = []
    
    for ref in references[:10]:  # Limit to 10 most relevant references
        ref_list.append({
            "url": ref.get("url"),
            "source": ref.get("source"),
            "tags": ref.get("tags", [])
        })
    
    return ref_list


def extract_cpe_summary(cve_item):
    """Extract affected product information from CPE configurations"""
    configurations = cve_item.get("configurations", [])
    
    if not configurations:
        return "No specific product configuration data available"
    
    # Extract unique product/vendor combinations
    products = set()
    
    for config in configurations:
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if cpe_match.get("vulnerable", False):
                    cpe_uri = cpe_match.get("criteria", "")
                    # Parse CPE format: cpe:2.3:a:vendor:product:version:...
                    parts = cpe_uri.split(":")
                    if len(parts) >= 5:
                        vendor = parts[3]
                        product = parts[4]
                        version_info = cpe_match.get("versionEndIncluding") or \
                                     cpe_match.get("versionEndExcluding") or \
                                     cpe_match.get("versionStartIncluding") or \
                                     parts[5] if len(parts) > 5 else ""
                        
                        products.add(f"{vendor}:{product} ({version_info})" if version_info else f"{vendor}:{product}")
    
    return list(products) if products else "Product details not parsed"


def validate_api_response(result):
    """Validate that critical fields are present"""
    warnings = []
    
    if not result.get("description") or result["description"] == "No description available":
        warnings.append("WARNING: CVE description missing - recommend HTML page verification")
    
    if not result.get("cvss_v3") and not result.get("cvss_v2"):
        warnings.append("WARNING: No CVSS scores available - CVE may be reserved/rejected")
    
    if result.get("cpe_configurations") == "No specific product configuration data available":
        warnings.append("INFO: No CPE configuration data - affected products may need manual research")
    
    return warnings


def main(cve_id, api_key=None):
    """Main execution function"""
    
    result = fetch_nvd_cve(cve_id, api_key)
    
    if not result:
        error_output = {
            "error": "CVE_NOT_FOUND",
            "message": f"CVE {cve_id} not found in NVD database",
            "cve_id": cve_id,
            "recommendation": "Verify CVE identifier is correct or check if CVE is reserved/rejected"
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)
    
    # Validate response and add warnings if needed
    warnings = validate_api_response(result)
    if warnings:
        result["warnings"] = warnings
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NVD CVE lookup - Fetch structured CVE data from National Vulnerability Database"
    )
    parser.add_argument("cve_id", help="CVE ID (e.g., CVE-2025-15467)")
    parser.add_argument(
        "--api-key", 
        help="NVD API key (optional, increases rate limit from 5 to 50 requests/30s)",
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        api_key = args.api_key()
        
        if not api_key:
            print("INFO: No NVD API key configured. Rate limit: 5 requests/30 seconds", file=sys.stderr)
            print("INFO: Add 'nvd.api_key' to config.yaml for 50 requests/30 seconds", file=sys.stderr)
        
        main(args.cve_id, api_key)
        
    except Exception as e:
        error_output = {
            "error": "SCRIPT_ERROR",
            "message": str(e),
            "cve_id": args.cve_id
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)
