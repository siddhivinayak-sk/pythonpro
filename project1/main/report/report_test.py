import os
import json
from types import SimpleNamespace

"""
Generate summary based on json file
"""


def generate_detail_for_entry(entry):
    obj = json.loads(entry, object_hook=lambda d: SimpleNamespace(**d))
    headers = obj.headers
    return Entry(
        get_key_value_from_list(headers, "applicationModuleCode"),
        get_key_value_from_list(headers, "applicationServiceCode"),
        obj.clientId if hasattr(obj, "clientId") else None,
        get_key_value_from_list(headers, "tokenType"),
        get_key_value_from_list(headers, "tenantid"),
    )


def get_key_value_from_list(list, key):
    for item in list:
        if item.key == key:
            return item.value
    return None


def generate_summary(json_file):
    print(f"JSON File: {json_file}")
    with open(json_file, "r") as file:
        json_data = file.readlines()
        for raw_entry in json_data:
            entry = generate_detail_for_entry(raw_entry)
            if entry.service_name == "":
                print(f"Entry: {entry}")


def generate_report(root_directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                generate_summary(json_file)


class Entry:
    def __init__(self, service_name, event_name, client_id, token_type, tenant_id):
        self.service_name = service_name
        self.event_name = event_name
        self.client_id = client_id
        self.token_type = token_type
        self.tenant_id = tenant_id

    def __str__(self):
        return f"service_name: {self.service_name}, event_name: {self.event_name}, client_id: {self.client_id}, token_type: {self.token_type}, tenant_id: {self.tenant_id}"


if __name__ == "__main__":
    directory = ""
    generate_report(directory)
