import urllib.request

url = "https://host.gitlab.com/api/v4/projects/0000212300/repository/files/some-file.xml/raw?ref=master"
token = "glpat-xxx"

headers = {
    'Private-Token': token,
}
request = urllib.request.Request(url, headers=headers)

try:
    response = urllib.request.urlopen(request)
    print("Request successful")
    print(f"Status code: {response.status}")
    content = response.read()
    print(f"Content length: {len(content)}")
    print(f"Content preview: {content[:500]}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
    print(f"Token authentication may have failed")
except urllib.error.URLError as e:
    print(f"Error: {e}")
