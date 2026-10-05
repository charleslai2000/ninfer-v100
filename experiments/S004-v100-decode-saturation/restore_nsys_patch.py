#!/usr/bin/env python3
"""Remove only the two env-gated S4 NVTX blocks from a disposable SGLang file."""
import sys
from pathlib import Path
p=Path(sys.argv[1]); s=p.read_text()
assert s.count('import torch.cuda.nvtx as s4_nvtx\n') == 1
s=s.replace('import torch.cuda.nvtx as s4_nvtx\n','',1)
a='''                        s4_capture = (
                            os.getenv("SGLANG_S4_CAPTURE_RANGE") == "1"
                            and batch.forward_mode.is_decode()
                            and batch.batch_size() == int(os.getenv("SGLANG_S4_CAPTURE_BS", "96"))
                        )
                        if s4_capture:
                            s4_range = s4_nvtx.range_start("S4_DECODE_RANGE")
                        try:
                            batch_result = self.model_worker.forward_batch_generation(
                                batch, **fwd_kwargs
                            )
                        finally:
                            if s4_capture:
                                s4_nvtx.range_end(s4_range)
'''
b='''                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
'''
assert s.count(a)==1;s=s.replace(a,b,1)
a='''                s4_capture = (
                    os.getenv("SGLANG_S4_CAPTURE_RANGE") == "1"
                    and batch.forward_mode.is_decode()
                    and batch.batch_size() == int(os.getenv("SGLANG_S4_CAPTURE_BS", "96"))
                )
                if s4_capture:
                    s4_range = s4_nvtx.range_start("S4_DECODE_RANGE")
                try:
                    batch_result = self.model_worker.forward_batch_generation(
                        batch, **kwargs
                    )
                finally:
                    if s4_capture:
                        s4_nvtx.range_end(s4_range)
'''
b='''                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
'''
assert s.count(a)==1;s=s.replace(a,b,1)
assert 'SGLANG_S4_CAPTURE_RANGE' not in s
p.write_text(s)
print("restored NVTX-only changes in",p)
