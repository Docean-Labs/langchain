"""Lightweight wrapper around requests library, with async support."""
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
import requests
from pydantic import BaseModel, Extra
import tiktoken


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(message, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(message, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        return num_tokens_from_messages(message, model="gpt-4-0314")
    num_tokens = 0
    num_tokens += len(encoding.encode(message))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _prep_output(response):
    if 'text/html' not in response.headers.get('Content-Type', ""):
        return response.text
    else:
        return "Not found any relevant information, please check your params and URL / Endpoint."


async def _aprep_output(response):
    if 'text/html' not in response.headers.get('Content-Type', ""):
        return await response.text()
    else:
        return "Not found any relevant information, please check your params and URL / Endpoint."


class Requests(BaseModel):
    """Wrapper around requests to handle auth and async.

    The main purpose of this wrapper is to handle authentication (by saving
    headers) and enable easy async methods on the same base object.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """GET the URL and return the text."""
        return requests.get(url, headers=self.headers, **kwargs)

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """POST to the URL and return the text."""
        return requests.post(url, json=data, headers=self.headers, **kwargs)

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PATCH the URL and return the text."""
        return requests.patch(url, json=data, headers=self.headers, **kwargs)

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PUT the URL and return the text."""
        return requests.put(url, json=data, headers=self.headers, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """DELETE the URL and return the text."""
        return requests.delete(url, headers=self.headers, **kwargs)

    @asynccontextmanager
    async def _arequest(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """Make an async request."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=self.headers, **kwargs
                ) as response:
                    yield response
        else:
            async with self.aiosession.request(
                method, url, headers=self.headers, **kwargs
            ) as response:
                yield response

    @asynccontextmanager
    async def aget(
        self, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """GET the URL and return the text asynchronously."""
        async with self._arequest("GET", url, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def apost(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """POST to the URL and return the text asynchronously."""
        async with self._arequest("POST", url, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def apatch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """PATCH the URL and return the text asynchronously."""
        async with self._arequest("PATCH", url, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def aput(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """PUT the URL and return the text asynchronously."""
        async with self._arequest("PUT", url, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def adelete(
        self, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """DELETE the URL and return the text asynchronously."""
        async with self._arequest("DELETE", url, **kwargs) as response:
            yield response


class TextRequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library.

    The main purpose of this wrapper is to always return a text output.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def requests(self) -> Requests:
        return Requests(headers=self.headers, aiosession=self.aiosession)

    def get(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text."""
        return _prep_output(self.requests.get(url, **kwargs))

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text."""
        return _prep_output(self.requests.post(url, data, **kwargs))

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text."""
        return _prep_output(self.requests.patch(url, data, **kwargs))

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text."""
        return _prep_output(self.requests.put(url, data, **kwargs))

    def delete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text."""
        return _prep_output(self.requests.delete(url, **kwargs))

    async def aget(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text asynchronously."""
        response = await self.requests.aget(url, **kwargs)
        return await _aprep_output(response)

    async def apost(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text asynchronously."""
        response = await self.requests.apost(url, data, **kwargs)
        return await _aprep_output(response)

    async def apatch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text asynchronously."""
        response = await self.requests.apatch(url, data, **kwargs)
        return await _aprep_output(response)

    async def aput(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text asynchronously."""
        response = await self.requests.aput(url, data, **kwargs)
        return await _aprep_output(response)

    async def adelete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text asynchronously."""
        response = await self.requests.adelete(url, **kwargs)
        return await _aprep_output(response)

    async def apost(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text asynchronously."""
        async with self.requests.apost(url, **kwargs) as response:
            return await response.text()

    async def apatch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text asynchronously."""
        async with self.requests.apatch(url, **kwargs) as response:
            return await response.text()

    async def aput(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text asynchronously."""
        async with self.requests.aput(url, **kwargs) as response:
            return await response.text()

    async def adelete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text asynchronously."""
        async with self.requests.adelete(url, **kwargs) as response:
            return await response.text()


# For backwards compatibility
RequestsWrapper = TextRequestsWrapper
