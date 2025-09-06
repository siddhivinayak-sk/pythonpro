from dotenv import load_dotenv
import os
from litellm import completion

load_dotenv()

os.environ["AZURE_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
os.environ["AZURE_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

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
    #print(f'Cache Creation Input Tokens: {usage["cache_creation_input_tokens"]}')
    #print(f'Cache Read Input Tokens: {usage["cache_read_input_tokens"]}')
    prompt_token_detail = usage["prompt_tokens_details"]
    print(f'Prompt Tokens Details - Audio Tokens: {prompt_token_detail.audio_tokens}')
    print(f'Prompt Tokens Details - Cached Tokens: {prompt_token_detail.cached_tokens}')
    print(f'Prompt Tokens Details - Text Tokens: {prompt_token_detail.text_tokens}')
    print(f'Prompt Tokens Details - Image Tokens: {prompt_token_detail.image_tokens}')

response = completion(
    model = "azure/" + deployment, 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
        timeout=1800,
        user="user",
        temperature=0.7,
        top_p=1,
        max_tokens=1000,
)

parse_response(response)