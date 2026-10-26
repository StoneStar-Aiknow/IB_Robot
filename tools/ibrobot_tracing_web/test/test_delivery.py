"""Verify transfer negotiation and deploy-safe SPA cache behavior."""

import gzip

import httpx

from ibrobot_tracing_web.app import create_app
from ibrobot_tracing_web.config import WebConfig
from ibrobot_tracing_web.delivery import DeliveryMiddleware


def test_compressed_assets_preserve_bytes_and_navigation_revalidates(tmp_path):
    import asyncio

    async def run():
        web = tmp_path / "web"
        assets = web / "assets"
        assets.mkdir(parents=True)
        payload = "const trace = 'request';\n" * 3000
        (assets / "index-ABcd1234.js").write_text(payload)
        (web / "index.html").write_text("<main>Trace</main>")
        (web / "plain.js").write_text(payload)
        traces = tmp_path / "traces"
        traces.mkdir()
        app = create_app(WebConfig(trace_roots=(traces,), allowed_hosts=("testserver",), web_root=web))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            identity = await client.get("/assets/index-ABcd1234.js", headers={"Accept-Encoding": "identity"})
            compressed = await client.get("/assets/index-ABcd1234.js", headers={"Accept-Encoding": "gzip"})
            assert identity.text == compressed.text == payload
            assert "content-encoding" not in identity.headers
            assert compressed.headers["content-encoding"] == "gzip"
            assert "Accept-Encoding" in compressed.headers["vary"]
            async with client.stream("GET", "/assets/index-ABcd1234.js", headers={"Accept-Encoding": "gzip"}) as stream:
                wire = b"".join([chunk async for chunk in stream.aiter_raw()])
            assert gzip.decompress(wire) == identity.content
            assert len(wire) < len(identity.content) // 4
            assert "immutable" in compressed.headers["cache-control"]
            for route in ("/", "/guide/capture", "/analysis/old-id", "/index.html", "/plain.js"):
                response = await client.get(route)
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-cache"
            missing = await client.get("/assets/missing-ABcd1234.js")
            assert missing.status_code == 404
            assert "immutable" not in missing.headers.get("cache-control", "")
            assert (await client.get("/api/missing")).status_code == 404
            for encoding in ("gzip;q=0", "gzip;q=0, *;q=1", "notgzip", "gzip;q=invalid"):
                rejected = await client.get("/plain.js", headers={"Accept-Encoding": encoding})
                assert rejected.content == identity.content and "content-encoding" not in rejected.headers
            assert identity.headers["etag"].startswith('"')
            assert compressed.headers["etag"] == "W/" + identity.headers["etag"]
            ranged = await client.get("/plain.js", headers={"Accept-Encoding": "gzip", "Range": "bytes=100-999"})
            assert ranged.status_code == 206 and ranged.content == identity.content[100:1000]
            assert "content-encoding" not in ranged.headers
            assert ranged.headers["content-range"] == f"bytes 100-999/{len(identity.content)}"
            head = await client.head("/plain.js", headers={"Accept-Encoding": "gzip"})
            assert not head.content and "content-encoding" not in head.headers
            html = await client.get("/")
            for condition in (
                {"If-None-Match": html.headers["etag"]},
                {"If-Modified-Since": html.headers["last-modified"]},
            ):
                conditional = await client.get("/guide/capture", headers=condition)
                assert conditional.status_code == 304 and not conditional.content
            changed = await client.get(
                "/", headers={"If-None-Match": '"wrong"', "If-Modified-Since": html.headers["last-modified"]}
            )
            assert changed.status_code == 200

            async def partial_app(scope, receive, send):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 206,
                        "headers": [(b"content-range", b"bytes 0-999/10000")],
                    }
                )
                await send({"type": "http.response.body", "body": b"a" * 1000})

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=DeliveryMiddleware(partial_app)), base_url="http://testserver"
            ) as partial:
                response = await partial.get("/", headers={"Accept-Encoding": "gzip"})
                assert response.content == b"a" * 1000 and "content-encoding" not in response.headers

    asyncio.run(run())
