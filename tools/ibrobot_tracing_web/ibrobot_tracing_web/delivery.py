"""HTTP representation negotiation around Starlette's existing gzip/file code."""

import re
from contextlib import suppress
from email.utils import parsedate_to_datetime

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import FileResponse, Response


def accepts_gzip(value: str) -> bool:
    qualities = {}
    for item in value.split(","):
        parts = [part.strip().lower() for part in item.split(";")]
        quality = 1.0
        for parameter in parts[1:]:
            if parameter.startswith("q="):
                raw = parameter[2:]
                quality = float(raw) if re.fullmatch(r"(?:0(?:\.\d{0,3})?|1(?:\.0{0,3})?)", raw) else 0.0
        # Duplicate declarations fail closed rather than overriding q=0.
        qualities[parts[0]] = min(quality, qualities.get(parts[0], 1.0))
    return qualities.get("gzip", qualities.get("*", 0.0)) > 0


class DeliveryMiddleware:
    def __init__(self, app):
        self.app = app
        self.gzip = GZipMiddleware(self._partial_safe, minimum_size=500, compresslevel=5)

    async def _partial_safe(self, scope, receive, send):
        async def protect(message):
            if message["type"] == "http.response.start" and message["status"] == 206:
                # Older Starlette versions lack their own 206 exclusion. Mark
                # the identity representation so gzip passes through unchanged.
                headers = MutableHeaders(scope=message)
                if "content-encoding" not in headers:
                    headers["content-encoding"] = "identity"
            await send(message)

        await self.app(scope, receive, protect)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        compress = (
            accepts_gzip(headers.get("accept-encoding", "")) and "range" not in headers and scope["method"] != "HEAD"
        )

        async def deliver(message):
            if message["type"] == "http.response.start":
                outgoing = MutableHeaders(scope=message)
                outgoing.add_vary_header("Accept-Encoding")
                if outgoing.get("content-encoding") == "identity":
                    del outgoing["content-encoding"]
                if outgoing.get("content-encoding") == "gzip" or (compress and message["status"] == 304):
                    etag = outgoing.get("etag")
                    if etag and not etag.startswith("W/"):
                        outgoing["etag"] = "W/" + etag
            await send(message)

        if compress:
            normalized = dict(scope)
            normalized["headers"] = [
                (key, value) for key, value in scope["headers"] if key.lower() != b"accept-encoding"
            ] + [(b"accept-encoding", b"gzip")]
            await self.gzip(normalized, receive, deliver)
        else:
            await self.app(scope, receive, deliver)


def static_response(path, request, cache_control):
    stat = path.stat()
    response = FileResponse(path, stat_result=stat, headers={"Cache-Control": cache_control})
    etag = response.headers["etag"]
    condition = request.headers.get("if-none-match")
    not_modified = False
    if condition is not None:
        not_modified = any(tag.strip().removeprefix("W/") in ("*", etag) for tag in condition.split(","))
    elif request.headers.get("if-modified-since"):
        with suppress(TypeError, ValueError, OverflowError):
            not_modified = int(stat.st_mtime) <= parsedate_to_datetime(request.headers["if-modified-since"]).timestamp()
    if not_modified:
        return Response(
            status_code=304, headers={key: response.headers[key] for key in ("etag", "last-modified", "cache-control")}
        )
    return response
