import asyncio
import httpx
import time

async def test_httpx_timeout():
    try:
        async with httpx.AsyncClient() as client:
            await client.get("http://httpbin.org/delay/2", timeout=1.0)
    except Exception as e:
        print(f"Exception type: {type(e)}")
        print(f"Connection Error: {str(e)}")

asyncio.run(test_httpx_timeout())
