"""Opt-in diagnostic worker; stream waits here are not production policy."""
import cProfile
import functools
import json
import pstats
import time

import torch

from vllm.v1.worker.gpu_worker import Worker


class WarmupDiagnosticWorker(Worker):
    def compile_or_warm_up_model(self):
        runner = self.model_runner
        originals = {}

        def wrap(name, original):
            def measured(iteration, *args, **kwargs):
                from shm_manager.torch import model_info
                manager = model_info.manager
                stream = torch.cuda.current_stream(self.device)
                before = manager.completed_layer_load_seconds()
                start = time.perf_counter()
                stream.synchronize()
                prior_wait = time.perf_counter() - start
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                profiler = cProfile.Profile()
                start = time.perf_counter()
                profiler.enable()
                try:
                    result = original(*args, **kwargs)
                finally:
                    profiler.disable()
                submitted = time.perf_counter() - start
                end.record(stream)
                start = time.perf_counter()
                end.synchronize()
                tail_wait = time.perf_counter() - start
                after = manager.completed_layer_load_seconds()
                stats = pstats.Stats(profiler)
                hot = sorted(stats.stats.items(), key=lambda pair: pair[1][3],
                             reverse=True)[:15]
                print('WARMUP_DIAGNOSTIC ' + json.dumps(dict(
                    rank=self.rank, phase=name, iteration=iteration,
                    prior_compute_wait=prior_wait,
                    cpu_submission_seconds=submitted,
                    compute_tail_wait_seconds=tail_wait,
                    # The stream interval includes host launch gaps; it is
                    # not the sum of GPU kernel execution durations.
                    compute_stream_span_seconds=begin.elapsed_time(end) / 1000,
                    completed_layers_before=len(before),
                    completed_layers_after=len(after),
                    hot_calls=[dict(file=key[0], line=key[1], function=key[2],
                                    calls=value[1], own_seconds=value[2],
                                    cumulative_seconds=value[3])
                               for key, value in hot])), flush=True)
                return result
            @functools.wraps(original)
            def repeated(*args, **kwargs):
                result = measured(0, *args, **kwargs)
                measured(1, *args, **kwargs)
                return result
            return repeated

        for name in ('_dummy_run', '_dummy_sampler_run'):
            originals[name] = getattr(runner, name)
            setattr(runner, name, wrap(name, originals[name]))
        try:
            return super().compile_or_warm_up_model()
        finally:
            for name, original in originals.items():
                setattr(runner, name, original)
