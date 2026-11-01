"""Hydra promotion policy driven by measured cold-start profiles.

Choose the smallest PP that meets TTFT/TPOT, as the upstream allocator does.
Use measured end-to-end cold TTFT rather than its fixed 2000 ms init term.
"""
from bisect import bisect_left
from copy import deepcopy
import math


class HydraPlanner:
    def __init__(self, profile):
        self.profile = deepcopy(profile)
        if profile.get('loading_policy') != 'serial':
            raise ValueError('Hydra requires a serial-loading profile; recalibrate legacy profiles')
        if (profile.get('version') != 1 or not profile.get('model')
                or type(profile.get('tp')) is not int or profile['tp'] < 1
                or type(profile.get('max_model_len')) is not int
                or profile['max_model_len'] < 1):
            raise ValueError('Invalid Hydra profile identity')
        profiles = self.profile.get('pp_profiles')
        native_blocks = self.profile.get('native_num_gpu_blocks')
        if native_blocks is not None and (type(native_blocks) is not int or native_blocks <= 0):
            raise ValueError('Hydra native cache capacity must be positive integer blocks')
        if not isinstance(profiles, list) or not profiles:
            raise ValueError('Hydra requires measured PP profiles')
        seen = set()
        for entry in profiles:
            pp = entry.get('pp')
            if type(pp) is not int or pp < 1 or pp in seen:
                raise ValueError('Hydra PP degrees must be positive and unique')
            seen.add(pp)
            phases = entry.get('phases')
            if not isinstance(phases, dict) or not phases or phases.keys() - {'first', 'reload'}:
                raise ValueError('Hydra requires explicit initialization phases')
            for samples in phases.values():
                if not isinstance(samples, list) or not samples:
                    raise ValueError('Hydra requires measured input samples')
                lengths = set()
                for row in samples:
                    length = row.get('input_tokens')
                    if (type(length) is not int or not 0 < length <= profile['max_model_len']
                            or length in lengths):
                        raise ValueError('Invalid or duplicated Hydra input length')
                    lengths.add(length)
                    for key in ('cold_ttft_ms', 'tpot_ms', 'ttft_error_ms'):
                        value = row.get(key, 0 if key == 'ttft_error_ms' else None)
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                            raise ValueError('Hydra timings must be finite nonnegative milliseconds')
                    if type(row.get('required_free_bytes')) is not int or row['required_free_bytes'] <= 0:
                        raise ValueError('Hydra requires measured per-rank memory budgets')
                samples.sort(key=lambda row: row['input_tokens'])
        if 1 not in seen:
            raise ValueError('Hydra requires a native PP=1 fallback profile')

    def candidates(self, input_tokens, *, ttft_slo_ms, tpot_slo_ms,
                   initialization_phase='first', placement_phases=None):
        if type(input_tokens) is not int or not 0 < input_tokens <= self.profile['max_model_len']:
            raise ValueError('Input length outside Hydra context')
        for value in (ttft_slo_ms, tpot_slo_ms):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Hydra SLOs must be finite positive milliseconds')
        if initialization_phase not in ('first', 'reload'):
            raise ValueError('Invalid Hydra initialization phase')
        if placement_phases is not None and (
                not isinstance(placement_phases, dict)
                or any(type(pp) is not int or pp < 1 or phase not in ('first', 'reload')
                       for pp, phase in placement_phases.items())):
            raise ValueError('Invalid Hydra placement phases')
        choices = []
        for entry in self.profile['pp_profiles']:
            if placement_phases is not None and entry['pp'] not in placement_phases:
                continue
            phase = (initialization_phase if placement_phases is None
                     else placement_phases[entry['pp']])
            samples = entry['phases'].get(phase, [])
            if not samples or not (
                    samples[0]['input_tokens'] <= input_tokens <=
                    samples[-1]['input_tokens']):
                continue
            index = bisect_left([row['input_tokens'] for row in samples], input_tokens)
            high = samples[index]
            low = high if high['input_tokens'] == input_tokens else samples[index - 1]
            ratio = (0 if low is high else (input_tokens - low['input_tokens']) /
                     (high['input_tokens'] - low['input_tokens']))
            interpolate = lambda key: low[key] + ratio * (high[key] - low[key])
            ttft = interpolate('cold_ttft_ms')
            upper = ttft + max(low.get('ttft_error_ms', 0), high.get('ttft_error_ms', 0))
            tpot = interpolate('tpot_ms')
            choices.append(dict(pp=entry['pp'], predicted_ttft_ms=ttft,
                ttft_upper_ms=upper, predicted_tpot_ms=tpot,
                required_free_bytes=max(low['required_free_bytes'], high['required_free_bytes']),
                slo_met=upper <= ttft_slo_ms and tpot <= tpot_slo_ms,
                input_tokens=input_tokens, initialization_phase=phase))
            # Reload measurements may reuse allocator storage present before
            # the sample. That saving is not transferable to two co-resident
            # models; use the measured envelope for overlap admission.
            choices[-1]['resident_required_free_bytes'] = max(
                row['required_free_bytes']
                for rows in entry['phases'].values() for row in rows)
        native = next((row for row in choices if row['pp'] == 1), None)
        if native is None:
            raise ValueError('Native Hydra profile does not cover this length and phase')
        for choice in choices:
            choice['recovery_required_free_bytes'] = native['resident_required_free_bytes']
            if self.profile.get('native_num_gpu_blocks') is not None:
                choice['native_num_gpu_blocks'] = self.profile['native_num_gpu_blocks']
        feasible = sorted((row for row in choices if row['slo_met']), key=lambda row: row['pp'])
        # If no admissible candidate meets SLO, try the largest measured PP
        # first. Capacity failures may then reduce PP, without labelling a
        # missed SLO as satisfied.
        fallback = sorted((row for row in choices if not row['slo_met']),
                          key=lambda row: row['pp'], reverse=True)
        return feasible + fallback
