"""Authenticated loopback service, warmed once per effective configuration."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import sys
import threading
import time
from .settings import Settings

MAX_LINE=2*1024*1024

def endpoint_path(root,settings):
    value=settings.effective()
    model=Path(root)/"models"/settings.model
    # A resident process is reusable only with the same inputs and executable
    # code. Include actual bytes, so replacing a bundle cannot reuse old weights.
    from .checkpoint import sha256
    value["model_files"]={name:sha256(model/name) for name in
                          ("conversion-manifest.json","config.json","actor.safetensors")}
    source=Path(__file__).resolve().parents[2]
    value["source_files"]={str(path.relative_to(source)):sha256(path)
                           for path in sorted(source.rglob("*.py"))}
    import importlib.util
    native=importlib.util.find_spec("libriichi")
    value["libriichi_sha256"]=sha256(native.origin) if native and native.origin else None
    value["python"]={"executable":str(Path(sys.executable).resolve()), "version":sys.version}
    key=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:20]
    return Path(root)/".runtime"/f"{key}.json"

def send(stream,value):
    stream.write(json.dumps(value,ensure_ascii=True,separators=(",",":"),allow_nan=False).encode()+b"\n")
    stream.flush()

def receive(stream):
    line=stream.readline(MAX_LINE+1)
    if not line: raise EOFError("Service connection closed")
    if len(line)>MAX_LINE: raise ValueError("MJAI message exceeds size limit")
    return json.loads(line)

def rpc(endpoint,kind):
    info=json.loads(Path(endpoint).read_text())
    with socket.create_connection(("127.0.0.1",info["port"]),timeout=5) as sock, sock.makefile("rwb") as stream:
        send(stream,{"token":info["token"],"type":kind})
        return receive(stream)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--settings",required=True)
    a=p.parse_args(); root=a.root.resolve()
    settings=Settings.parse(json.loads(a.settings)); settings.validate_qualification(root)
    endpoint=endpoint_path(root,settings); endpoint.parent.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock,Timeout
    lock=FileLock(str(endpoint)+".lock")
    try: lock.acquire(timeout=0)
    except Timeout: return 2
    try:
        # Imports and CUDA initialization happen only in the resident process.
        import numpy as np
        from .actor_torch import Predictor
        from .session import Session
        started=time.perf_counter(); effective=settings.effective()
        predictor=Predictor(root/"models"/settings.model,**{k:effective[k] for k in ("device","precision","attention","cpu_threads")})
        from .warmup import warmup
        for _ in range(settings.warmup): warmup(predictor)
        token=secrets.token_hex(32); infer_lock=threading.Lock()
        stats={"sessions":0,"decisions":0,"ready_seconds":time.perf_counter()-started,"model_load_count":1}
        ready={"status":"ready","pid":os.getpid(),"model":settings.model,"identity":predictor.manifest["actor_sha256"],"requested":json.loads(a.settings),"effective":{**effective,**predictor.effective},"stats":stats}
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    self.connection.settimeout(3600)
                    hello=receive(self.rfile)
                    if not isinstance(hello,dict) or not secrets.compare_digest(str(hello.get("token","")),token): raise ValueError("Invalid service token")
                    kind=hello.get("type")
                    if kind=="ping": send(self.wfile,ready); return
                    if kind=="stop":
                        send(self.wfile,{"status":"stopping"})
                        threading.Thread(target=self.server.shutdown,daemon=True).start(); return
                    if kind!="session" or hello.get("model")!=settings.model: raise ValueError("Invalid session model")
                    session=Session(predictor,hello["seat"])
                    with infer_lock: stats["sessions"]+=1
                    send(self.wfile,{"status":"ok","model":settings.model,"identity":ready["identity"],"effective":ready["effective"]})
                    while not session.ended:
                        events=receive(self.rfile)
                        started=time.perf_counter()
                        with infer_lock:
                            response=session.react(events)
                            stats["decisions"]+=bool(response.get("meta",{}).get("decision"))
                            stages=getattr(session.engine,"last_timings",{})
                        elapsed=(time.perf_counter()-started)*1000
                        if settings.diagnostics: print(json.dumps({"event":"reaction","model":settings.model,"milliseconds":elapsed,"stages":stages}),file=sys.stderr,flush=True)
                        send(self.wfile,response)
                except EOFError: pass
                except Exception as error:
                    try: send(self.wfile,{"status":"error","error":str(error)})
                    except OSError: pass
                    print(f"session error: {error}",file=sys.stderr,flush=True)
        class Server(socketserver.ThreadingTCPServer):
            daemon_threads=True
            allow_reuse_address=False
        with Server(("127.0.0.1",0),Handler) as server:
            record={"port":server.server_address[1],"token":token,"pid":os.getpid()}
            temp=endpoint.with_suffix(".tmp")
            temp.write_text(json.dumps(record),encoding="utf-8")
            temp.replace(endpoint)
            print(json.dumps(ready),file=sys.stderr,flush=True)
            try: server.serve_forever(poll_interval=.2)
            finally: endpoint.unlink(missing_ok=True)
    finally: lock.release()
    return 0
if __name__=="__main__": raise SystemExit(main())
