"""Apply the opt-in scheduler batch histogram to the disposable remote S1 extraction.
The hook is one in-memory integer increment per decode iteration and exports only
with the existing 40-step stats line / /server_info; no tensor or CUDA work.
"""
from pathlib import Path
root=Path('/hy-tmp/sglang-V100-s2-07343bd165/python/sglang/srt')
p=root/'environ.py'; s=p.read_text(); anchor='    SGLANG_RECORD_STEP_TIME = EnvBool(False)\n'; assert anchor in s and 'SGLANG_S4_BATCH_HISTOGRAM' not in s
p.write_text(s.replace(anchor,anchor+'    SGLANG_S4_BATCH_HISTOGRAM = EnvBool(False)\n',1))
p=root/'managers/scheduler_components/metrics_reporter.py'; s=p.read_text()
anchor='        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]\n'; assert anchor in s and 'decode_batch_hist' not in s
s=s.replace(anchor,anchor+'        self.decode_batch_hist = defaultdict(int)\n',1)
anchor='        batch = running_batch or self.scheduler.running_batch\n\n        # Every-iteration work: realtime token counting + status logger\n'; assert anchor in s
s=s.replace(anchor,'        batch = running_batch or self.scheduler.running_batch\n        if envs.SGLANG_S4_BATCH_HISTOGRAM.get():\n            self.decode_batch_hist[len(batch.reqs)] += 1\n\n        # Every-iteration work: realtime token counting + status logger\n',1)
anchor='''            f"#queue-req: {len(self.scheduler.waiting_queue)}"
        )
'''; assert anchor in s
s=s.replace(anchor,'''            f"#queue-req: {len(self.scheduler.waiting_queue)}, "
            f"decode_batch_hist={dict(self.decode_batch_hist)}"
        )
''',1);p.write_text(s)
p=root/'managers/scheduler.py'; s=p.read_text(); anchor='''        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.metrics_reporter.step_time_dict
'''; assert anchor in s and 'decode_batch_hist' not in s
s=s.replace(anchor,anchor+'''        if envs.SGLANG_S4_BATCH_HISTOGRAM.get():
            ret["decode_batch_hist"] = dict(self.metrics_reporter.decode_batch_hist)
''',1);p.write_text(s)
