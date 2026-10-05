#!/usr/bin/env python3
"""Sample NVIDIA SMI power, utilization, memory while a benchmark child runs."""
import csv, subprocess, sys, time
from pathlib import Path
out=Path(sys.argv[1]); cmd=sys.argv[2:]
if not str(out).startswith('/hy-tmp/sglang-logs/s4-v100-'): raise SystemExit('unsafe output path')
with out.open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['epoch_s','memory_used_mib','gpu_util_pct','power_w','temperature_c']); f.flush()
 p=subprocess.Popen(cmd)
 try:
  while p.poll() is None:
   r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   if r.returncode==0:
    vals=[x.strip() for x in r.stdout.splitlines()[0].split(',')]
    w.writerow([time.time(),*vals]); f.flush()
   time.sleep(.25)
 finally:
  rc=p.wait()
 raise SystemExit(rc)
