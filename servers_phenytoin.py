# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Launch/teardown the phenytoin generality-demo Tesseract servers.

The whole point of the demo: the CKD physiology Tesseract is the SAME served
module as the vancomycin engine (ckd_physiology), REUSED unchanged. Only the PK
Tesseract (phenytoin_pk, saturable) and the loss Tesseract (phenytoin_loss,
concentration target) are swapped. Same one-process-one-port pattern as servers.py.
"""
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(os.path.dirname(HERE), "venv", "Scripts")
RUNTIME_EXE = os.path.join(VENV, "tesseract-runtime.exe")

# ckd = the SAME module the vanco engine uses (reused, unchanged).
MODULES = {
    "ckd": ("ckd_physiology", 8041),
    "pk": ("phenytoin_pk", 8042),
    "loss": ("phenytoin_loss", 8043),
}


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
def serve_all():
    procs = []
    urls = {}
    try:
        for key, (folder, port) in MODULES.items():
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
        for key, (folder, port) in MODULES.items():
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
