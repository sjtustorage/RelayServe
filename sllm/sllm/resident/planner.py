"""Bind the existing analytic planner to a resident producer's real metadata."""
from copy import deepcopy
import math
import os

from sllm.sllm.loading_perf_profile import (
    _prepare_solver_inputs, _resolve_profile_model_key,
    _solve_lazy_load_method_cpp_with_ttft,
    _solve_lazy_load_method_cpp_with_ttft as _solve_gpu_only_cpp,
)


class ResidentPlanner:
    def __init__(self, profile, checkpoint, *, gpu_tp, cpu_tp, config,
                 max_model_len=None,
                 copy_throttle=True, copy_throttle_tasks=32,
                 copy_throttle_first_only=False):
        if type(copy_throttle_first_only) is not bool:
            raise ValueError('copy_throttle_first_only must be a boolean')
        if copy_throttle_first_only and not copy_throttle:
            raise ValueError('first-only throttle requires first-model throttling')
        self.copy_throttle_first_only = copy_throttle_first_only
        if (max_model_len is not None and
                (type(max_model_len) is not int or max_model_len < 1)):
            raise ValueError('max_model_len must be a positive integer')
        self.max_model_len = max_model_len
        self.reload_planner = (ResidentPlanner(profile, checkpoint, gpu_tp=gpu_tp,
            cpu_tp=cpu_tp, config=config, max_model_len=max_model_len, copy_throttle=False,
            copy_throttle_tasks=copy_throttle_tasks)
            if copy_throttle_first_only else None)
        if type(copy_throttle_tasks) is not int or copy_throttle_tasks < 1:
            raise ValueError('copy_throttle_tasks must be a positive integer')
        self.copy_throttle_tasks = copy_throttle_tasks
        if type(copy_throttle) is not bool:
            raise ValueError('copy_throttle must be a boolean')
        self.copy_throttle = copy_throttle
        self.profile = deepcopy(profile)
        self.checkpoint = str(checkpoint)
        self.backend = dict(tensor_parallel_size=gpu_tp, pipeline_parallel_size=1)
        self.config = dict(config)
        key = _resolve_profile_model_key(self.profile.get('models', {}), self.checkpoint)
        if key is None:
            raise ValueError('No planner profile for checkpoint')
        entry = self.profile['models'][key]
        context_profiles = entry.get('resident_context_profiles', {})
        if not isinstance(context_profiles, dict) or any(
                not isinstance(length, str) or not length.isdigit() or int(length) < 1
                for length in context_profiles):
            raise ValueError('Context profiles must be keyed by positive max_model_len')
        self.context_profile = None
        if max_model_len is not None:
            context = context_profiles.get(str(max_model_len))
            if context is not None:
                calibration = context.get(
                    'resident_context_initialization_calibration', {})
                expected_flash = (calibration.get('flash_attn_version_override')
                                  if isinstance(calibration, dict) else None)
                observed_flash = os.environ.get('VLLM_FLASH_ATTN_VERSION')
                # Context timings are backend-specific. Do not apply an FA3
                # calibration to another attention backend or an unrecorded
                # environment.
                if expected_flash is not None and observed_flash != str(expected_flash):
                    context = None
                # A different warmup workload changes the CPU overlap budget.
                # Compare recorded policy overrides, including an explicitly
                # recorded default (None), before using its initialization fit.
                if isinstance(calibration, dict):
                    if (calibration.get('initialization_window_origin')
                            != 'vllm_engine_core_entry'):
                        context = None
                    for field, env_name in (
                        ('relay_warmup_prefill_tokens', 'VLLM_RELAY_WARMUP_PREFILL_TOKENS'),
                        ('relay_first_warmup_prefill_tokens', 'VLLM_RELAY_FIRST_WARMUP_PREFILL_TOKENS'),
                        ('relay_initialization_target_ms', 'VLLM_RELAY_INITIALIZATION_TARGET_MS'),
                    ):
                        if field in calibration:
                            expected = calibration[field]
                            observed = os.environ.get(env_name)
                            if observed != (str(expected) if expected is not None else None):
                                context = None
            if context is not None:
                allowed = {'resident_initialization_ms',
                           'resident_startup_ms', 'resident_ttft_error_bounds_ms',
                           'resident_ttft_error_calibration',
                           'resident_copy_profiles',
                           'resident_context_initialization_calibration'}
                if not isinstance(context, dict) or not context or context.keys() - allowed:
                    raise ValueError('Context profile contains unsupported fields')
                # Context timing is a complete measured scope. In particular,
                # a global method/copy profile must not override its init time.
                for field in ('resident_initialization_ms',
                              'resident_startup_ms', 'resident_ttft_error_bounds_ms',
                              'resident_copy_profiles'):
                    entry.pop(field, None)
                entry.update(deepcopy(context))
                self.context_profile = str(max_model_len)
        mode = ('throttled' if copy_throttle_tasks == 32 else
                f'throttled_tasks_{copy_throttle_tasks}') if copy_throttle else 'unthrottled'
        variants = entry.get('resident_copy_profiles', {})
        if not isinstance(variants, dict) or any(
                key not in ('throttled', 'unthrottled') and
                not (isinstance(key, str) and key.startswith('throttled_tasks_')
                     and key.removeprefix('throttled_tasks_').isdigit()
                     and int(key.removeprefix('throttled_tasks_')) > 0)
                for key in variants):
            raise ValueError('Copy profiles must use throttled/unthrottled modes')
        if mode in variants:
            variant = variants[mode]
            if (not isinstance(variant, dict) or
                    not variant.get('resident_initialization_ms') or
                    variant.keys() - {'resident_initialization_ms',
                                      'resident_startup_ms',
                                      'resident_ttft_error_bounds_ms'}):
                raise ValueError('Copy mode requires measured initialization profiles')
            # Never inherit timings measured under the other copy policy.
            for field in ('resident_initialization_ms',
                          'resident_startup_ms', 'resident_ttft_error_bounds_ms'):
                entry.pop(field, None)
            entry.update(deepcopy(variant))
        elif not copy_throttle or copy_throttle_tasks != 32:
            raise ValueError('Unthrottled copy requires its own measured initialization profile')
        startup = entry.get('resident_startup_ms', {})
        if (not isinstance(startup, dict) or
                startup.keys() - {'gpu_create', 'weight_submit', 'cpu_dispatch'} or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in startup.values())):
            raise ValueError('Resident startup times must be finite nonnegative milliseconds')
        self.startup = dict(startup)
        # Positive relay residuals and negative GPU-only residuals both
        # reduce confidence in a predicted saving. Keep independent bounds
        # for the two initialization phases and execution methods.
        self.ttft_error_bounds = entry.get('resident_ttft_error_bounds_ms', {})
        if (not isinstance(self.ttft_error_bounds, dict)
                or self.ttft_error_bounds.keys() - {'first', 'reload'}):
            raise ValueError('TTFT error bounds must use first/reload phases')
        for bounds in self.ttft_error_bounds.values():
            if not isinstance(bounds, dict):
                raise ValueError('TTFT error bounds require a method mapping')
            by_input = bounds.get('per_input_length', {})
            if (bounds.keys() - {'tokenwise', 'layerwise', 'gpu_only', 'per_input_length'}
                    or not isinstance(by_input, dict)
                    or any(not isinstance(k, str) or not k.isdigit() or int(k) < 1
                           for k in by_input)):
                raise ValueError('TTFT error bounds require valid prompt lengths')
            rows = [{k: v for k, v in bounds.items() if k != 'per_input_length'},
                    *by_input.values()]
            if any(not isinstance(row, dict)
                    or row.keys() != {'tokenwise', 'layerwise', 'gpu_only'}
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                           for v in row.values()) for row in rows):
                raise ValueError('TTFT error bounds require finite nonnegative method bounds')
        self.initialization_profiles = entry.get('resident_initialization_ms', {})
        measurements = list(self.initialization_profiles.values()) if isinstance(
            self.initialization_profiles, dict) else []
        if (not isinstance(self.initialization_profiles, dict)
                or self.initialization_profiles.keys() - {'first', 'reload'}):
            raise ValueError('Resident initialization profiles must use first/reload phases')
        for measurement in measurements:
            if (not isinstance(measurement, dict)
                    or measurement.keys() != {'engine_init_ms', 'resident_startup_ms'}):
                raise ValueError('Each initialization phase requires lazy and startup measurements')
            lazy = measurement['engine_init_ms']
            offsets = measurement['resident_startup_ms']
            if (type(lazy) not in (int, float) or not math.isfinite(lazy) or lazy < 0
                    or not isinstance(offsets, dict)
                    or offsets.keys() != {'gpu_create', 'weight_submit', 'cpu_dispatch'}
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                           for v in offsets.values())):
                raise ValueError('Initialization measurements must be finite nonnegative milliseconds')
        if entry.get('tp_size') != gpu_tp or entry.get('pp_size', 1) != 1:
            raise ValueError('Planner GPU profile does not match resident topology')
        if entry.get('cpu_tp_size', 2) != cpu_tp:
            raise ValueError('Planner CPU profile does not match producer topology')
        required = ('num_hidden_layers', 'hidden_size', 'num_attention_heads',
                    'num_key_value_heads')
        if any(type(self.config.get(key)) is not int or self.config[key] <= 0 for key in required):
            raise ValueError('Planner requires real model architecture metadata')

    def __call__(self, prompt_length, *, initialization_phase='first',
                 max_num_batched_tokens=4096):
        if self.reload_planner is not None and initialization_phase == 'reload':
            return self.reload_planner(prompt_length, initialization_phase='reload',
                                       max_num_batched_tokens=max_num_batched_tokens)
        if type(max_num_batched_tokens) is not int or max_num_batched_tokens < 1:
            raise ValueError('Planner chunk budget must be a positive integer')
        if initialization_phase not in ('first', 'reload'):
            raise ValueError('Unknown resident initialization phase')
        inputs = _prepare_solver_inputs(self.profile, self.checkpoint, prompt_length,
                                        backend_config=self.backend)
        measurement = self.initialization_profiles.get(initialization_phase)
        if (not self.copy_throttle or self.copy_throttle_tasks != 32) and measurement is None:
            raise ValueError('Missing unthrottled initialization measurement for phase ' + initialization_phase)
        startup = self.startup
        if measurement is not None:
            inputs['fixed_ms'] = measurement['engine_init_ms']
            startup = measurement['resident_startup_ms']
        config = self.config
        inputs.update(num_layers=config['num_hidden_layers'], hidden_size=config['hidden_size'],
                      num_kv_heads=config['num_key_value_heads'],
                      head_dim=config.get('head_dim', config['hidden_size'] // config['num_attention_heads']))
        # Tokenwise assists only the first prefill chunk and emits at most
        # one sample; layerwise continues according to GPU boundary feedback.
        max_cpu_tokens = min(prompt_length, max_num_batched_tokens)
        inputs.update(max_cpu_tokens=max_cpu_tokens,
                      allow_cpu_decode=False, prefill_chunk_tokens=max_num_batched_tokens)
        # CPU begins after admission, before model-slot creation finishes.
        # DMA starts in lazy initialization, after model-slot creation and
        # native allocation/submission. fixed_ms is lazy duration in profiles
        # that explicitly supply these resident startup measurements.
        create = startup.get('gpu_create', 0.0)
        inputs.update(cpu_start_ms=startup.get('cpu_dispatch', 0.0),
                      weight_start_ms=create + startup.get('weight_submit', 0.0),
                      fixed_ms=inputs['fixed_ms'] + create)
        solved = _solve_lazy_load_method_cpp_with_ttft(inputs, backend_config=self.backend)
        if solved is None or len(solved) != 4:
            raise ValueError('Planner returned no complete plan')
        method, amount, partitions, predicted = solved
        total = config['num_hidden_layers']
        full_cpu = method == 'layerwise' and amount == total
        if full_cpu:
            # All layers on CPU produce a complete prefill/sample. Transfer
            # its KV through tokenwise continuation rather than initialize an
            # empty GPU suffix or publish final logits as hidden states.
            method, amount, partitions = 'tokenwise', prompt_length, [list(range(total))]
        if (method not in ('tokenwise', 'layerwise') or type(amount) is not int
                or amount < 0 or (method == 'layerwise' and amount >= total)
                or len(partitions) != 1 or predicted is None
                or not math.isfinite(predicted) or predicted < 0):
            raise ValueError('Planner returned invalid resident plan')
        # The solver may append prefix recovery layers to the load order.
        # Execution is the contiguous suffix; the loader owns reverse recovery.
        layers = list(range(amount if method == 'layerwise' else 0, total))
        # Tokenwise partitions describe the first weight batch, not the full
        # model execution range. The asynchronous loader also submits its tail.
        if method == 'tokenwise' and partitions[0] != list(range(len(partitions[0]))):
            raise ValueError('Planner tokenwise preload must be a contiguous prefix')
        if any(type(layer) is not int or not 0 <= layer < total for layer in partitions[0]):
            raise ValueError('Planner partition contains invalid layers')
        if method == 'layerwise' and not set(layers) <= set(partitions[0]):
            raise ValueError('Planner partition omits execution layers')
        # Use the same native cost model for the zero-CPU comparison.
        baseline_inputs = deepcopy(inputs)
        baseline_inputs['max_cpu_tokens'] = 0
        baseline = _solve_gpu_only_cpp(
            baseline_inputs, backend_config=self.backend)
        baseline_predicted = float(baseline[3])
        bounds = self.ttft_error_bounds.get(initialization_phase, {})
        bounds = bounds.get('per_input_length', {}).get(str(prompt_length), bounds)
        relay_upper = predicted + bounds.get(method, 0.0)
        baseline_lower = max(0.0, baseline_predicted - bounds.get('gpu_only', 0.0))
        selected_gpu_only = relay_upper >= baseline_lower
        if selected_gpu_only:
            method, amount, predicted = 'tokenwise', 0, baseline_predicted
            full_cpu = False
            layers = list(range(total))
        prefix = amount if method == 'tokenwise' else 0
        if prefix > inputs['max_cpu_tokens']:
            raise ValueError('Planner CPU prefix exceeds the executable first chunk')
        return dict(method=method, amount=amount, full_cpu_prefill=full_cpu, execution_layers=layers,
                    initialization_phase=initialization_phase,
                    copy_throttle=self.copy_throttle,
                    copy_throttle_tasks=self.copy_throttle_tasks,
                    initialization_profile_phase=initialization_phase if measurement is not None else 'legacy',
                    initialization_profile_method='shared',
                    context_profile=self.context_profile,
                    max_model_len=self.max_model_len,
                    initialization_ready_ms=inputs['fixed_ms'],
                    weight_start_ms=inputs['weight_start_ms'],
                    predicted_ttft_ms=predicted,
                    baseline_predicted_ttft_ms=(float(baseline_predicted)
                                                if baseline_predicted is not None else None),
                    candidate_ttft_upper_ms=relay_upper,
                    baseline_ttft_lower_ms=baseline_lower,
                    benefit_error_bounds_applied=bool(bounds),
                    selected_gpu_only=selected_gpu_only,
                    prediction_model='intra_node_equations_2_5',
                    cpu_prefix_tokens=prefix,
                    execution_partition_validated=True,
                    cpu_layers=amount if method == 'layerwise' else 0,
                    max_num_batched_tokens=max_num_batched_tokens,
                    input_length=prompt_length, planned_input_length=inputs['input_length'])
