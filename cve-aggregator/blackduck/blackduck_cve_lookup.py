import requests
import argparse
import sys
import json
from pathlib import Path

BASE_URL = "https://my-company.app.blackduck.com"
AUTH_URL = f"{BASE_URL}/api/tokens/authenticate"
SEARCH_URL = f"{BASE_URL}/api/search/vulnerabilities"

    
def get_bearer_token(api_key):
    HEADERS_AUTH = {
    "Authorization": f"token {api_key}"
    }
    response = requests.post(AUTH_URL, headers=HEADERS_AUTH)
    response.raise_for_status()
    return response.json()["bearerToken"]


def search_vulnerability(bearer_token, cve_id):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {"q": cve_id}

    response = requests.get(SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def extract_vulnerability_items(search_response, input_cve):
    bdsa_item = None
    nvd_item = None

    for item in search_response.get("items", []):
        source = item.get("vulnerabilitySource")

        if (
            source == "BDSA"
            and item.get("relatedVulnerabilityId") == input_cve
        ):
            bdsa_item = item

        elif source == "NVD":
            nvd_item = item

    return bdsa_item, nvd_item



def get_affected_bom_components_paginated(bearer_token, base_url):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    limit = 100
    offset = 0
    all_items = []

    while True:
        params = {
            "limit": limit,
            "offset": offset
        }

        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])
        total_count = data.get("_meta", {}).get("totalCount", 0)

        all_items.extend(items)

        offset += limit
        if offset >= total_count:
            break

    return all_items



def collect_projects(items):
    projects = set()
    for item in items:
        project_name = item.get("projectName")
        if project_name:
            projects.add(project_name)
    return sorted(projects)



def main(cve_id, api_key=None):
    bearer_token = get_bearer_token(api_key)

    search_response = search_vulnerability(bearer_token, cve_id)
    bdsa_item, nvd_item = extract_vulnerability_items(search_response, cve_id)

    if not bdsa_item:
        print(f"No BDSA vulnerability found for {cve_id}", file=sys.stderr)
        sys.exit(1)

    affected_bom_url = None
    for link in bdsa_item.get("_meta", {}).get("links", []):
        if link.get("rel") == "affected-bom-components":
            affected_bom_url = link.get("href")
            break

    if not affected_bom_url:
        print("affected-bom-components link not found", file=sys.stderr)
        sys.exit(1)

    bdsa_vulnerabilities_url = None
    bdsa_vulnerabilities_url = bdsa_item.get("_meta", {}).get("href", "")

    affected_response = get_affected_bom_components_paginated(bearer_token, affected_bom_url)
    projects = collect_projects(affected_response)

    final_output = {
    "vulnerabilitySource": "BDSA",
    "bdsa_vulnerabilities_url": bdsa_vulnerabilities_url,
    "vulnerabilityId": bdsa_item.get("vulnerabilityId"),
    "relatedVulnerabilityId": bdsa_item.get("relatedVulnerabilityId"),
    "bdsa": {
        "vulnerabilitySeverityType": bdsa_item.get("vulnerabilitySeverityType"),
        "overallScore": bdsa_item.get("overallScore"),
        "summary": bdsa_item.get("summary")
    },
    "nvd": {
        "vulnerabilitySeverityType": nvd_item.get("vulnerabilitySeverityType") if nvd_item else None,
        "overallScore": nvd_item.get("overallScore") if nvd_item else None
    },
    "affectedProjects": projects
    }

    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlackDuck CVE lookup")
    parser.add_argument("cve_id", help="CVE ID (e.g. CVE-2024-33599)")
    args = parser.parse_args()
    api_key = None

    try:
        main(args.cve_id, api_key)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
