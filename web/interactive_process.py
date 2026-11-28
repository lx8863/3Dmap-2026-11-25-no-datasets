"""Bounded line protocol for the isolated CUDA renderer process."""

import base64
import binascii
import json
from queue import Empty, Full, Queue
import subprocess
from threading import Thread


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
            start_new_session=True,
        )
        self.responses = Queue(maxsize=1)
        Thread(target=self.read_lines, daemon=True).start()

    def read_lines(self):
        # Windows does not support select.poll() on anonymous subprocess pipes.
        while True:
            try:
                line = self.process.stdout.readline(MAX_RESPONSE_BYTES + 1)
                if not line:
                    raise InteractiveRenderError("交互渲染器进程已退出。")
                if len(line) > MAX_RESPONSE_BYTES:
                    raise InteractiveRenderError("交互渲染器响应过大。")
                if not line.endswith(b"\n"):
                    raise InteractiveRenderError("交互渲染器响应不完整。")
                response = line
            except (InteractiveRenderError, OSError, ValueError) as exc:
                response = InteractiveRenderError(str(exc))
            try:
                self.responses.put(response, timeout=RENDER_FRAME_TIMEOUT)
            except Full:
                return
            if isinstance(response, InteractiveRenderError):
                return

    def read_response(self, timeout):
        try:
            line = self.responses.get(timeout=timeout)
        except Empty as exc:
            raise InteractiveRenderError("交互渲染器响应超时。") from exc
        if isinstance(line, InteractiveRenderError):
            raise line
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
        self.process.stdin.flush()
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
