"""Bounded line protocol for the isolated CUDA renderer process."""

import base64
import binascii
import json
import os
import select
import subprocess
import time


MAX_RESPONSE_BYTES = 2 * 1024 * 1024
RENDER_START_TIMEOUT = 30
RENDER_FRAME_TIMEOUT = 10


class InteractiveRenderError(RuntimeError):
    pass


class InteractiveProcess:
    def __init__(self, command, cwd, environment):
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        os.set_blocking(self.process.stdin.fileno(), False)
        self.buffer = bytearray()

    def read_response(self, timeout):
        deadline = time.monotonic() + timeout
        descriptor = self.process.stdout.fileno()
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = self.buffer[:newline]
                del self.buffer[:newline + 1]
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not poller.poll(max(1, int(remaining * 1000))):
                raise InteractiveRenderError("交互渲染器响应超时。")
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                raise InteractiveRenderError("交互渲染器进程已退出。")
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_RESPONSE_BYTES:
                raise InteractiveRenderError("交互渲染器响应过大。")
        try:
            response = json.loads(line)
        except (ValueError, RecursionError) as exc:
            raise InteractiveRenderError("交互渲染器返回了无效的响应。") from exc
        if not isinstance(response, dict):
            raise InteractiveRenderError("交互渲染器返回了无效的响应。")
        if isinstance(response.get("error"), str):
            raise InteractiveRenderError(response["error"][:500])
        return response

    def wait_until_ready(self):
        if self.read_response(RENDER_START_TIMEOUT).get("ready") is not True:
            raise InteractiveRenderError("交互渲染器未能启动。")

    def render(self, pose):
        request = json.dumps(pose, separators=(",", ":")).encode("utf-8") + b"\n"
        if self.process.stdin.write(request) != len(request):
            raise InteractiveRenderError("交互渲染器未能接受请求。")
        encoded = self.read_response(RENDER_FRAME_TIMEOUT).get("jpeg")
        if not isinstance(encoded, str):
            raise InteractiveRenderError("交互渲染器未返回 JPEG。")
        try:
            jpeg = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise InteractiveRenderError("交互渲染器返回了无效的 JPEG。") from exc
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise InteractiveRenderError("交互渲染器返回了无效的 JPEG。")
        return jpeg
