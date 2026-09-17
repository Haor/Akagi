"""Standard-library client and native service lifecycle; never invokes WSL."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from .service import endpoint_path,rpc,send,receive
from .settings import Settings

def ensure_service(root,settings,timeout=120):
    root=Path(root).resolve()
    if not (root/"models"/settings.model/"conversion-manifest.json").is_file():
        raise ValueError(f"Model is not installed: {settings.model}; export it to models/{settings.model}")
    endpoint=endpoint_path(root,settings)
    try:
        value=rpc(endpoint,"ping")
        if value.get("status")=="ready": return endpoint
    except (OSError,ValueError,EOFError): pass
    endpoint.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-m","rin.inference.native.service","--root",str(root),"--settings",json.dumps(asdict(settings))]
    log=endpoint.with_suffix(".log")
    env=os.environ.copy()
    env["PYTHONPATH"]=str(Path(__file__).resolve().parents[3])
    with log.open("ab") as stream:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,env=env,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            value=rpc(endpoint,"ping")
            if value.get("status")=="ready": return endpoint
        except (OSError,ValueError,EOFError): pass
        if process.poll() not in (None,2):
            raise RuntimeError(f"Native service failed ({process.returncode}): {log.read_text(errors='replace')[-3000:]}")
        time.sleep(.1)
    raise TimeoutError(f"Native service did not become ready; see {log}")

class Connection:
    def __init__(self,endpoint,model,seat):
        info=json.loads(Path(endpoint).read_text())
        self.socket=socket.create_connection(("127.0.0.1",info["port"]),timeout=30)
        self.stream=self.socket.makefile("rwb")
        send(self.stream,{"token":info["token"],"type":"session","model":model,"seat":seat})
        self.info=receive(self.stream)
        if self.info.get("status")!="ok": self.close(); raise RuntimeError(self.info)
    def react(self,events):
        send(self.stream,events)
        value=receive(self.stream)
        if value.get("status")=="error": raise RuntimeError(value["error"])
        return value
    def close(self): self.stream.close(); self.socket.close()

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--settings",type=Path)
    p.add_argument("--action",choices=["start","check","stop","stop-all","bot"],default="bot")
    p.add_argument("--seat",type=int,default=0)
    a=p.parse_args()
    if a.action=="stop-all":
        for endpoint in (a.root/".runtime").glob("*.json"):
            try: print(endpoint.name,json.dumps(rpc(endpoint,"stop")))
            except (OSError,ValueError,EOFError) as error: print(f"{endpoint.name}: {error}",file=sys.stderr)
        return
    value=json.loads(a.settings.read_text(encoding="utf-8-sig")) if a.settings else {}
    settings=Settings.parse(value); settings.validate_qualification(a.root)
    endpoint=endpoint_path(a.root,settings)
    if a.action=="stop": print(json.dumps(rpc(endpoint,"stop"))); return
    if a.action=="check": print(json.dumps(rpc(endpoint,"ping"))); return
    endpoint=ensure_service(a.root,settings)
    if a.action=="start": print(json.dumps(rpc(endpoint,"ping"))); return
    conn=Connection(endpoint,settings.model,a.seat)
    print(json.dumps({"model_identity":conn.info}),file=sys.stderr,flush=True)
    try:
        for line in sys.stdin:
            if not line.strip(): continue
            events=json.loads(line)
            response=conn.react(events)
            print(json.dumps(response,ensure_ascii=True,separators=(",",":")),flush=True)
            if any(e.get("type")=="end_game" for e in events): break
    finally: conn.close()
if __name__=="__main__":
    try: main()
    except Exception as error:
        print(f"RIN native error: {error}",file=sys.stderr,flush=True)
        raise SystemExit(1)
