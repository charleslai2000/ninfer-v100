#!/usr/bin/env python3
"""Create deterministic T018 hidden rows matching the F6C2C layer-input scale/seed family."""
import argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('layer',type=int);p.add_argument('tokens',type=int);a=p.parse_args()
seed=0xF6C2C + a.layer*1009 + a.tokens
rng=np.random.default_rng(seed)
x=(0.2*(rng.random((a.tokens,3072),dtype=np.float32)-0.5)).astype('<f4')
a.output.parent.mkdir(parents=True,exist_ok=True);x.tofile(a.output)
print(f'FIXTURE layer={a.layer} T={a.tokens} seed={seed} bytes={x.nbytes} output={a.output}')
