"""V1 CPU control messages shared by resident TP/PP scheduler replicas."""
import torch.distributed as dist

from vllm.distributed import get_pp_group, get_tp_group
from vllm.v1.engine.core import EngineCore


class ResidentEngineCore(EngineCore):
    def recv_shm_scheduler_output(self):
        group = get_tp_group()
        pipeline = get_pp_group()
        message = [None]
        if group.is_first_rank and pipeline.is_first_rank:
            message[0] = super().recv_shm_scheduler_output()
        # First distribute within stage zero, then down each TP lane. Only
        # one rank consumes the CPU semaphore for the entire model instance.
        dist.broadcast_object_list(message, src=group.first_rank,
                                   group=group.cpu_group)
        if pipeline.world_size > 1:
            dist.broadcast_object_list(message, src=pipeline.first_rank,
                                       group=pipeline.cpu_group)
        return message[0]

    def send_shm_model_output(self, model_output):
        # One producer/consumer semaphore pair belongs to the whole instance.
        # TP schedulers use identical feedback, but only the leader publishes.
        if get_tp_group().is_first_rank and get_pp_group().is_first_rank:
            super().send_shm_model_output(model_output)
        elif (getattr(self, '_next_cpu_boundary', None) == 0
              or self.loaded_layers == self.target_layers):
            # CPU release follows the agreed execution boundary. A forecast
            # may release CPU before every weight has completed; forward
            # fences each layer separately. All TP schedulers must stop
            # receiving CPU batches at the same iteration as the leader.
            self.need_send_shm_model_output = False
