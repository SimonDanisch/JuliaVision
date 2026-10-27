"""Serve exactly one public checkpoint on a chosen interface, with resume support.

The receiver must still verify the pinned SHA256. No directory listing or writes
are exposed. Run only during the requested cross-machine model transfer.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


class CheckpointHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        super().__init__(*args, **kwargs)

    def send_head(self):
        if unquote(urlsplit(self.path).path) != "/" + self.checkpoint.name:
            self.send_error(404)
            return None
        size = self.checkpoint.stat().st_size
        requested = self.headers.get("Range")
        start = 0
        if requested:
            match = re.fullmatch(r"bytes=(\d+)-", requested)
            if not match or int(match[1]) >= size:
                self.send_error(416)
                return None
            start = int(match[1])
        stream = self.checkpoint.open("rb")
        stream.seek(start)
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size - start))
        self.send_header("Accept-Ranges", "bytes")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.end_headers()
        return stream


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    path = args.checkpoint.resolve(strict=True)
    server = ThreadingHTTPServer((args.bind, args.port),
                                 partial(CheckpointHandler, checkpoint=path))
    print(f"Serving only {path.name} on {args.bind}:{args.port}", flush=True)
    server.serve_forever()
