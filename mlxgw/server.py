"""Starlette app: /v1/chat/completions, /v1/models, auth, OpenAI error shape."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import logging
import os
import sys
import threading
import time

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .admin import Admin, QuietPolling, admin_routes
from .api import APIError, ChatRequest, chunk, completion, invalid, new_id, parse_chat_request, tool_call, usage
from .config import Settings, load_settings, log_profile
from .engine import Final, ToolCall

log = logging.getLogger("mlxgw")


def check_auth_config(settings: Settings) -> None:
    if not settings.is_loopback and not settings.api_key:
        raise SystemExit(f"refusing to bind {settings.host}: non-loopback addresses require an API key "
                         "(set MLXGW_API_KEY or server.api_key)")


def _error_response(e: APIError) -> JSONResponse:
    return JSONResponse(e.body(), status_code=e.status)


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def create_app(engine, settings: Settings, admin: Admin | None = None) -> Starlette:
    """`engine` provides aliases() -> {alias: repo}, check_model(alias) and
    stream(ChatRequest, cancel_event) -> async iterator of str deltas and ToolCalls, then one Final.
    For /admin it also provides resident() -> {repo: bytes}, async load(alias),
    async unload(alias) -> bool and worker.busy()."""
    check_auth_config(settings)
    admin = admin or Admin(settings, engine)

    def authorize(request: Request) -> None:
        if not settings.api_key:
            return
        expected = f"Bearer {settings.api_key}".encode()
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), expected):
            raise APIError(401, "invalid or missing API key", code="invalid_api_key")

    async def models(request: Request) -> JSONResponse:
        authorize(request)
        data = [{"id": a, "object": "model", "created": 0, "owned_by": "mlxgw", "x_repo": repo}
                for a, repo in engine.aliases().items()]
        return JSONResponse({"object": "list", "data": data})

    async def chat(request: Request):
        authorize(request)
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise invalid("request body is not valid JSON", code="invalid_json")
        req = parse_chat_request(body)
        engine.check_model(req.model)
        admin.check_not_downloading(req.model)
        cancel = threading.Event()
        if req.stream:
            return StreamingResponse(stream_chat(req, cancel), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        watcher = asyncio.create_task(cancel_on_disconnect(request, cancel))
        try:
            parts, calls, final = [], [], None
            async for item in engine.stream(req, cancel):
                if isinstance(item, Final):
                    final = item
                elif isinstance(item, ToolCall):
                    calls.append(item)
                else:
                    parts.append(item)
        except Exception as e:
            admin.record(req.model, None, getattr(e, "message", None) or type(e).__name__)
            raise
        finally:
            watcher.cancel()
        admin.record(req.model, final)
        u = usage(final.prompt_tokens, final.completion_tokens, final.cached_tokens)
        return JSONResponse(completion(new_id(), req.model, int(time.time()), "".join(parts),
                                       final.finish_reason, u, final.timings, calls))

    async def cancel_on_disconnect(request: Request, cancel: threading.Event) -> None:
        # Non-streaming only: the body has been read, so the next message is the disconnect.
        while (await request.receive())["type"] != "http.disconnect":
            pass
        log.info("client disconnected; cancelling generation")
        cancel.set()

    async def stream_chat(req: ChatRequest, cancel: threading.Event):
        cid, created = new_id(), int(time.time())
        extra = {"usage": None} if req.include_usage else {}
        try:
            yield _sse(chunk(cid, req.model, created, {"role": "assistant", "content": ""}, **extra))
            final, n_calls = None, 0
            async for item in engine.stream(req, cancel):
                if isinstance(item, Final):
                    final = item
                    admin.record(req.model, final)
                elif isinstance(item, ToolCall):  # sent whole, once its end marker is seen
                    yield _sse(chunk(cid, req.model, created, {"tool_calls": [tool_call(item, n_calls)]}, **extra))
                    n_calls += 1
                else:
                    yield _sse(chunk(cid, req.model, created, {"content": item}, **extra))
            last = chunk(cid, req.model, created, {}, final.finish_reason, **extra)
            if req.include_usage:
                yield _sse(last)
                u = usage(final.prompt_tokens, final.completion_tokens, final.cached_tokens)
                yield _sse(chunk(cid, req.model, created, None, usage=u, x_gateway=final.timings))
            else:
                yield _sse({**last, "x_gateway": final.timings})
        except APIError as e:
            admin.record(req.model, None, e.message)
            yield _sse(e.body())
        except Exception as e:
            admin.record(req.model, None, type(e).__name__)
            log.exception("generation failed")
            yield _sse(APIError(500, "internal error during generation", type="server_error").body())
        except asyncio.CancelledError:
            log.info("client disconnected mid-stream; cancelling generation")
            cancel.set()
            raise
        yield "data: [DONE]\n\n"

    async def api_error(_request, exc: APIError):
        return _error_response(exc)

    async def http_error(_request, exc: HTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code)
        return _error_response(APIError(exc.status_code, str(exc.detail), code=code))

    async def server_error(_request, exc: Exception):
        log.exception("unhandled error", exc_info=exc)
        return _error_response(APIError(500, "internal server error", type="server_error"))

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        # uvicorn configures its own loggers before startup, so attach the ring here.
        loggers = [logging.getLogger("mlxgw"), logging.getLogger("uvicorn")]
        access, quiet = logging.getLogger("uvicorn.access"), QuietPolling()
        for lg in loggers:
            lg.addHandler(admin.logs)
        access.addFilter(quiet)
        threading.Thread(target=admin.fetch_remote_sizes, name="remote-sizes", daemon=True).start()
        try:
            yield
        finally:
            admin.downloads.cancel_all()
            access.removeFilter(quiet)
            for lg in loggers:
                lg.removeHandler(admin.logs)

    return Starlette(
        routes=[Route("/v1/chat/completions", chat, methods=["POST"]),
                Route("/v1/models", models, methods=["GET"]),
                *admin_routes(admin, authorize)],
        exception_handlers={APIError: api_error, HTTPException: http_error, Exception: server_error},
        lifespan=lifespan,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Local OpenAI-compatible gateway for MLX models")
    ap.add_argument("--config", default=os.environ.get("MLXGW_CONFIG", "config.toml"))
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # huggingface_hub logs every file request

    settings = load_settings(args.config, host=args.host, port=args.port)
    log_profile(settings)
    try:
        check_auth_config(settings)
    except SystemExit as e:
        log.error("%s", e)
        sys.exit(2)

    import uvicorn
    from .engine import MLXEngine

    app = create_app(MLXEngine(settings), settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
