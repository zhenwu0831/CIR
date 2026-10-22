import asyncio
import os

import aiohttp

# Configuration
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")

# Set up headers for the request
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}


def format_message(messages, role, prompt):
    messages.append({"role": role, "content": [{"type": "text", "text": prompt}]})
    return messages


def create_llm_payload(messages, parameters=None):
    if parameters is None:
        parameters = {}
    payload = {
        "messages": messages,
        "temperature": parameters.get("temperature", 0.6),
        "max_tokens": parameters.get("max_tokens", 1000),
    }
    return payload


async def async_generator(payload, max_retries=5, backoff=5):
    """
    Asynchronous helper function to send a POST request to the LLM endpoint
    using aiohttp, with simple retry logic.

    Args:
        payload (dict): JSON payload for the POST request.
        max_retries (int): Number of times to retry on failure.
        backoff (int): Seconds to wait between retries.

    Returns:
        str: The 'content' from the LLM response.
    """
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(ENDPOINT, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    response_data = await response.json()

                    # Return the message's content from the first choice
                    return response_data["choices"][0]["message"]["content"]

        except Exception as e:
            # If we haven't exhausted retries, wait and try again
            if attempt < max_retries - 1:
                print(
                    f"[async_generator] Request failed (attempt {attempt + 1}). "
                    f"Retrying in {backoff} seconds..."
                )
                await asyncio.sleep(backoff)
            else:
                # If we've exhausted our retries, re-raise the error
                raise e


def generator(payload):
    """
    Synchronous wrapper that calls the async_generator using asyncio.run().
    This lets the rest of the code remain synchronous, but under the hood,
    the call is performed asynchronously with retries.
    """
    return asyncio.run(async_generator(payload=payload))
