import os
from litellm import completion

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def bedrock_env_setup():
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["MY_AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["MY_AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_REGION_NAME"] = os.environ["MY_AWS_REGION_NAME"]

def parse_response(response):
    print(f'Response Id: {response["id"]}')
    print(f'Create At: {response["created"]}')
    print(f'Model: {response["model"]}')
    print(f'Object: {response["object"]}')
    print(f'System Fingerprint: {response["system_fingerprint"]}')
    choices = response["choices"]
    for choice in choices:
        print(f'Choice Index: {choice["index"]}')
        print(f'Finish Reason: {choice["finish_reason"]}')
        message = choice["message"]
        print(f'Message Role: {message["role"]}')
        print(f'Message Content: {message["content"]}')
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                print(f'Tool Call Id: {tool_call["id"]}')
                print(f'Tool Call Function Name: {tool_call["function"]["name"]}')
                print(f'Tool Call Function Arguments: {tool_call["function"]["arguments"]}')
        print(f'Function Call: {message["function_call"]}')
        print(f'Provider Specific Fields: {message["provider_specific_fields"]}')
    usage = response["usage"]
    print(f'Usage Prompt Tokens: {usage["prompt_tokens"]}')
    print(f'Usage Completion Tokens: {usage["completion_tokens"]}')
    print(f'Total Tokens: {usage["total_tokens"]}')
    print(f'Completion Tokens Details: {usage["completion_tokens_details"]}')
    print(f'Cache Creation Input Tokens: {usage["cache_creation_input_tokens"]}')
    print(f'Cache Read Input Tokens: {usage["cache_read_input_tokens"]}')
    prompt_token_detail = usage["prompt_tokens_details"]
    print(f'Prompt Tokens Details - Audio Tokens: {prompt_token_detail.audio_tokens}')
    print(f'Prompt Tokens Details - Cached Tokens: {prompt_token_detail.cached_tokens}')
    print(f'Prompt Tokens Details - Text Tokens: {prompt_token_detail.text_tokens}')
    print(f'Prompt Tokens Details - Image Tokens: {prompt_token_detail.image_tokens}')

def llm_call():
    bedrock_env_setup()
    response = completion(
        model="bedrock/mistral.mistral-large-2402-v1:0",
        messages=[{'content': 'What is the capital of France?', 'role': 'user'}],
        # timeout=1800,
        user="user",
        # temperature=0.7,
        # top_p=1,
        # max_tokens=1000,
        # top_k=1,
        # m_tools=tools,
        # m_tool_choice='auto',
        # messages=[
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": "What is in this image?"},
        #             {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": "data:image/jpeg;base64," + encode_image('../data/test.png')
        #                 },
        #             },
        #         ],
        #     }
        # ],
        # reasoning_effort='low',
    )
    parse_response(response)


if __name__ == "__main__":
    llm_call()
