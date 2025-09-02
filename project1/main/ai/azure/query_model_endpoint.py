from typing_extensions import Self
from typing import TypedDict
from openai import AzureOpenAI


class QueryModelEndpoint:
    def __init__(self: Self, env: dict, search_config: dict) -> None:
        self.env = env
        self.search_config = search_config
        print(self.env)

    class Response(TypedDict):
        query: str
        response: str

    # @trace
    def __call__(self: Self, query: str) -> Response:
        client = AzureOpenAI(
            azure_endpoint=self.env["azure_endpoint"],
            api_version="2024-06-01",
            api_key=self.env["api_key"],
        )
        # Call the model
        completion = client.chat.completions.create(
            model=self.env["azure_deployment"],
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
            extra_body={
                "data_sources": [{
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": f"{self.search_config['search_endpoint']}",
                        "index_name": f"{self.search_config['search_index']}",
                        "semantic_configuration": "default",
                        "query_type": "semantic",
                        "fields_mapping": {},
                        "in_scope": True,
                        "filter": None,
                        "strictness": 3,
                        "top_n_documents": 5,
                        "authentication": {
                            "type": "api_key",
                            "key": f"{self.search_config['search_key']}"
                        }
                    }
                }]
            }
        )
        output = completion.to_dict()
        return {"query": query, "response": output["choices"][0]["message"]["content"]}