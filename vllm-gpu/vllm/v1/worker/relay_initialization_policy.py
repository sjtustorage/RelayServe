"""Explicit readiness windows for resident SHM model initialization."""
import math
import os


def initialization_target_ms(config):
    override = os.environ.get('VLLM_RELAY_INITIALIZATION_TARGET_MS')
    if override is not None:
        try:
            value = float(override)
        except ValueError as exc:
            raise ValueError('Initialization target must be finite and nonnegative') from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError('Initialization target must be finite and nonnegative')
        return value
    # Architecture identity survives the producer's temporary checkpoint path.
    return {(36, 4096): 700., (40, 5120): 700.,
            (64, 5120): 850., (80, 8192): 1000.}.get(
                (getattr(config, 'num_hidden_layers', None),
                 getattr(config, 'hidden_size', None)), 0.)


def readiness_wait_seconds(started, now, target_ms, serial_copy_seconds=0.):
    # A serial baseline pays transfer in addition to the same initialization
    # window. Counting its blocking copy inside the window would hide transfer
    # in the baseline and invalidate the comparison.
    return max(0., target_ms / 1000. + serial_copy_seconds - (now - started))
