import requests
import sys
import os

PROJECT_ID = "208719"
REQUIRED_LABEL = "status::reviewed"

    
def search_issues(search_string, access_token, gitlab_base_url):
    HEADERS = {
        "PRIVATE-TOKEN": access_token
    }
    page = 1
    per_page = 100

    result = {
        "issue_links": [],
        "descriptions": []
    }

    while True:
        url = f"{gitlab_base_url}/projects/{PROJECT_ID}/issues"
        params = {
            "scope": "all",
            "per_page": per_page,
            "page": page
        }

        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()

        issues = response.json()
        if not issues:
            break
        
        for issue in issues:
            labels = issue.get("labels", [])

            # ✅ only include reviewed issues
            if REQUIRED_LABEL not in labels:
                continue

            title = issue.get("title", "") or ""

            if search_string.lower() in title.lower():
                # ✅ full issue URL
                result["issue_links"].append(issue.get("web_url"))
                result["descriptions"].append(issue.get("description", "") or "")

        page += 1

    return result


if __name__ == "__main__":
    gitlab_url = os.getenv('GITLAB_API_V4')
    gitlab_token = os.getenv('GITLAB_API_TOKEN')
    try:
        result = search_issues("CVE-2025-31133", gitlab_token, gitlab_url)
        print(result)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        sys.exit(1)