#!/usr/bin/env python3
"""Patch disposable SGLang extraction with an env-gated NVTX range around one decode forward."""
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()
needle='import torch\n'
assert s.count(needle)==1
assert 'SGLANG_S4_CAPTURE_RANGE' not in s
s=s.replace(needle,needle+'import torch.cuda.nvtx as s4_nvtx\n',1)
needle='''                        # FIXME: pp is not compatible with overlap
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
'''
replacement='''                        # FIXME: pp is not compatible with overlap
                        s4_capture = (
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
assert s.count(needle)==1
s=s.replace(needle,replacement,1)
needle='''                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
'''
replacement='''                s4_capture = (
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
assert s.count(needle)==1
s=s.replace(needle,replacement,1)
p.write_text(s)
print("patched",p,"forward sites=2, opt-in gate=SGLANG_S4_CAPTURE_RANGE")
