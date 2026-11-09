"""Use orjson for FastAPI request bodies without changing body validation."""

from typing import Any

import orjson
from fastapi import Request
from fastapi.routing import APIRoute


class ORJSONRequest(Request):
    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            self._json = orjson.loads(await self.body())
        return self._json


class ORJSONRoute(APIRoute):
    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def custom_handler(request: Request):
            return await original_handler(ORJSONRequest(request.scope, request.receive))

        return custom_handler
