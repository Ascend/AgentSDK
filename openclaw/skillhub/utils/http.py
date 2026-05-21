"""HTTP client wrapper for API requests."""

from typing import Any, Dict, Optional
import httpx


class HttpClient:
    """Async HTTP client wrapper."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        http2: bool = True,
    ):
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or {},
            timeout=timeout,
            http2=http2,
        )

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        response = await self._client.get(path, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    async def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        response = await self._client.post(path, json=json, data=data, headers=headers)
        response.raise_for_status()
        return response.json()

    async def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        response = await self._client.put(path, json=json, headers=headers)
        response.raise_for_status()
        return response.json()

    async def delete(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        response = await self._client.delete(path, headers=headers)
        response.raise_for_status()
        return response.json()

    async def download(self, url: str, destination: str) -> str:
        import os

        response = await self._client.get(url, follow_redirects=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as f:
            f.write(response.content)
        return destination

    async def close(self):
        await self._client.aclose()
