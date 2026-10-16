import asyncio
import random
import sys
from copilot import CopilotClient, PermissionHandler
from copilot.generated.session_events import SessionEventType, SessionEvent
from copilot.tools import define_tool
from pydantic import BaseModel, Field

# Ref: https://github.com/github/awesome-copilot/blob/main/instructions/copilot-sdk-python.instructions.md


class GetWeatherParams(BaseModel):
    city: str = Field(description="The name of the city to get weather for")


@define_tool(description="Get the current weather for a city")
async def get_weather(params: GetWeatherParams) -> dict:
    city = params.city
    # Replace with actual weather API call
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    temp = random.randint(50, 80)
    condition = random.choice(conditions)
    return {"city": city, "temperature": f"{temp}°F", "condition": condition}


async def start_copilot():
    client = CopilotClient()
    await client.start()
    return client


async def stop_copilot(client):
    await client.stop()


async def create_session(client):
    session = await client.create_session({
        "model": "gpt-4.1",
        "on_permission_request": PermissionHandler.approve_all,
        "tools": [get_weather],
    })
    return session


async def create_session_stream(client):
    session = await client.create_session({
        "model": "gpt-4.1",
        "on_permission_request": PermissionHandler.approve_all,
        "stream": True,
        "tools": [get_weather],
        "mcpServers": { "github": { "type": "http", "url": "https://api.githubcopilot.com/mcp/"}},
        "customAgents": [{"name": "pr-reviewer", "displayName": "PR Reviewer", "description": "Reviews pull requests for best practices", "prompt": "You are an expert code reviewer. Focus on security, performance, and maintainability."}],
        "systemMessage": {"content": "You are a helpful assistant for our engineering team. Always be concise."}
    })
    return session


async def create_session_stream_with_event(client):
    session = await client.create_session({
        "model": "gpt-4.1",
        "on_permission_request": lambda req, inv: {"kind": "approved"},
        "stream": True,
        "tools": [get_weather],
    })
    return session


async def send_query(session, query):
    return await session.send_and_wait({"prompt": query})


async def simple_query(query: str):
    try:
        client = await start_copilot()
        session = await create_session(client)
        response = await send_query(session, query)
        await stop_copilot(client)
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


async def stream_query(query: str):
    try:
        client = await start_copilot()
        session = await create_session_stream(client)
        def handle_event(event): # Listen for response chunks
            if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                sys.stdout.write(event.data.delta_content)
                sys.stdout.flush()
            if event.type == SessionEventType.SESSION_IDLE:
                print()  # New line when done
        session.on(handle_event)
        response = await send_query(session, query)
        await stop_copilot(client)
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


async def stream_query_with_event(query: str):
    try:
        client = await start_copilot()
        session = await create_session_stream_with_event(client)
        unsubscribe = session.on(lambda event: print(f"Event: {event.type}"))
        # Filter by event type in your handler
        def handle_event(event: SessionEvent) -> None:
            if event.type == SessionEventType.SESSION_IDLE:
                print("Session is idle")
            elif event.type == SessionEventType.ASSISTANT_MESSAGE:
                print(f"Message: {event.data.content}")

        unsubscribe = session.on(handle_event)
        response = await send_query(session, query)
        await stop_copilot(client)
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


async def simple_run():
    query_text = "Use get_weather tool to answer: What's the weather like in Seattle and Tokyo?"
    # response = await simple_query(query_text)
    response = await stream_query(query_text)
    #response = await stream_query_with_event(query_text)
    print(response.data.content)


async def intractive():
    try:
        client = await start_copilot()
        session = await create_session_stream(client)
        
        def handle_event(event):
            if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                sys.stdout.write(event.data.delta_content)
                sys.stdout.flush()
        session.on(handle_event)

        print("🌤️  Weather Assistant (type 'exit' to quit)")
        print("   Try: 'What's the weather in Paris?' or 'Compare weather in NYC and LA'\n")

        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break
            if user_input.lower() == "exit":
                break
            sys.stdout.write("Assistant: ")
            response = await session.send_and_wait({"prompt": user_input})
            print(f"{response.data.content}\n")
        await stop_copilot(client)
    except Exception as e:
        print(f"An error occurred: {e}")


async def main():
    # await simple_run()
    await intractive()


asyncio.run(main())