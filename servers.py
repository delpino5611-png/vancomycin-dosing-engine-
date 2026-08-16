# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Launch/teardown the three vanco Tesseract servers (Docker-free tesseract-runtime).

One Tesseract = one process = one port (feasibility.md landmine #4). This context
manager starts all three, waits for /health, yields the from_url handles, and
guarantees teardown (avoids WinError 10048 zombie ports on re-runs).
"""
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(os.path.dirname(HERE), "venv", "Scripts")
RUNTIME_EXE = os.path.join(VENV, "tesseract-runtime.exe")

MODULES = {
    "ckd": ("ckd_physiology", 8031),
    "pk": ("vanco_pk", 8032),
    "loss": ("exposure_loss", 8033),
}

# Optional stretch module: the 2-compartment PK path. Served alongside the core
# when include_2c=True; the 1-comp entry never depends on it.
MODULE_2C = {"pk2c": ("vanco_pk_2c", 8034)}


def _healthy(port, timeout=0.5):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _wait(port, name, deadline=90.0):
    t0 = time.time()
    while time.time() - t0 < deadline:
        if _healthy(port):
            return True
        time.sleep(0.5)
    raise RuntimeError(f"server '{name}' on port {port} did not become healthy in {deadline}s")


@contextmanager
def serve_all(include_2c=False):
    modules = dict(MODULES)
    if include_2c:
        modules.update(MODULE_2C)
    procs = []
    urls = {}
    try:
        for key, (folder, port) in modules.items():
            api = os.path.join(HERE, folder, "tesseract_api.py")
            env = dict(os.environ, TESSERACT_API_PATH=api)
            log = open(os.path.join(HERE, folder, "serve.log"), "w")
            p = subprocess.Popen(
                [RUNTIME_EXE, "serve", "--port", str(port)],
                cwd=os.path.join(HERE, folder),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            procs.append((p, log))
            urls[key] = f"http://127.0.0.1:{port}"
        for key, (folder, port) in modules.items():
            _wait(port, folder)
            print(f"[serve] {folder} healthy on {urls[key]}")
        yield urls
    finally:
        for p, log in procs:
            try:
                p.terminate()
            except Exception:
                pass
            try:
                log.close()
            except Exception:
                pass
        for p, _ in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("[serve] all servers torn down")
