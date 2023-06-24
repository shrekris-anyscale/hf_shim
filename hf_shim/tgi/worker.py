import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type
from unittest.mock import patch

import torch
import torch.distributed
from filelock import FileLock
from text_generation_server.pb.generate_pb2 import (
    Request as GenerationRequest,
)

if TYPE_CHECKING:
    from text_generation_server.models.causal_lm import CausalLMBatch
    from text_generation_server.models.model import Model
    from text_generation_server.models.types import (
        Generation,
    )

_batch_id = 0


def _reset_batch_id():
    global _batch_id
    _batch_id = 0


def get_batch_id() -> int:
    global _batch_id
    _batch_id += 1
    return _batch_id


@dataclass
class FakePB2:
    requests: List[GenerationRequest]
    id: Optional[int] = None
    request_ids: Optional[List[int]] = None
    max_tokens: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self.requests) if self.requests else 0


def create_batch(
    model: "Model", requests: List["GenerationRequest"]
) -> Type["CausalLMBatch"]:
    return model.batch_type.from_pb(
        FakePB2(id=get_batch_id(), requests=requests),
        tokenizer=model.tokenizer,
        dtype=model.dtype,
        device=model.device,
    )


def concatenate_batches(
    model: "Model", batches: List["CausalLMBatch"]
) -> "CausalLMBatch":
    # Reset batch_id
    batches[0].batch_id = get_batch_id()
    return model.batch_type.concatenate(batches)


class InferenceWorker:
    def __init__(self, model_loader: Callable[[], "Model"]):
        self._model = model_loader()
        self._batch_state_cache: Dict[int, "CausalLMBatch"] = dict()
        if self._model.device.type == "cuda":
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    def process_new_batch(
        self, requests: List["GenerationRequest"]
    ) -> Tuple[List["Generation"], int]:
        batch_state = create_batch(self._model, requests)
        generations, batch_state = self._model.generate_token(batch_state)
        try:
            print("Batch state ID: ", batch_state.batch_id)
        except Exception as e:
            print("Failed to get batch state id", repr(e))
        if batch_state is not None:
            self._batch_state_cache[batch_state.batch_id] = batch_state
            return generations, batch_state.batch_id
        else:
            return generations, None

    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[List["Generation"], Optional[int]]:
        if len(batch_ids) == 0:
            raise ValueError("Must provide at least one batch")
        batch_states = []
        for batch_id in batch_ids:
            if batch_id is None:
                continue
            batch_state = self._batch_state_cache.pop(batch_id, None)
            if batch_state is None:
                raise ValueError(f"Batch ID {batch_id} not found in cache.")
            batch_states.append(batch_state)

        if len(batch_states) == 0:
            return [], None

        if len(batch_states) > 1:
            batch_state = concatenate_batches(self._model, batch_states)
        else:
            batch_state = batch_states[0]

        try:
            # stats = batch_state.stats()
            generations, batch_state = self._model.generate_token(batch_state)
        except Exception as e:
            print(f"error happened: {e}")
            #  Error happens when populate the new batch, we have to restart
            self._batch_state_cache.clear()
            return None, None

        if batch_state:
            self._batch_state_cache[batch_state.batch_id] = batch_state
            return generations, batch_state.batch_id
        return generations, None

    def filter_requests(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        if batch_id is None:
            return None

        batch_state = self._batch_state_cache.pop(batch_id)

        if len(request_ids) == 0:
            return None

        filtered = batch_state.filter(request_ids)
        # Reset batch_id

        filtered.batch_id = get_batch_id()
        if len(filtered):
            self._batch_state_cache[filtered.batch_id] = filtered
            return filtered.batch_id

        return None

    def report_stats(self):
        # print(f"worker stats: {[(id, cache.stats()) for id, cache in self._batch_state_cache.items()]}")
        if self._model.device.type == "cuda":
            # gc.collect()
            print(
                f"memory allocated: {torch.cuda.memory_allocated(self._model.device) / 2 ** 30}"
            )
            print(
                f"memory reserved: {torch.cuda.memory_reserved(self._model.device) / 2 ** 30}"
            )
            # self.check_cuda_objects()
            # if torch.cuda.memory_allocated(self._model.device) / 2 ** 30 > 30:
            #    self.debug_objects()

    def check_cuda_objects(self):
        from collections import defaultdict

        if self._model.device.type != "cuda":
            return
        d = defaultdict(int)

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    t = tuple(obj.size()) + (obj.dtype, obj.device)
                    d[t] += 1
            except Exception:
                pass

        for count, obj_signature in sorted(
            [(count, sig) for sig, count in d.items()], key=lambda x: x[0], reverse=True
        ):
            print(count, obj_signature)

    def debug_objects(self):
        objs = gc.get_objects()
        tensors = [obj for obj in objs if torch.is_tensor(obj)]
        leaked_tensors = [t for t in tensors if t.size() == torch.Size([20, 1, 1024])]
        if len(leaked_tensors) >= 1000:
            import pdb

            pdb.set_trace()


def initialize_torch_distributed():
    print("initialize_torch_distributed")
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    print(torch.distributed.get_world_size(), torch.cuda.device_count())

    from text_generation_server.utils.dist import FakeGroup

    if world_size == 1:
        return FakeGroup(rank, world_size), rank, world_size
    return torch.distributed.distributed_c10d._get_default_group(), rank, world_size


class TGIInferenceWorker(InferenceWorker):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str],
        sharded: Optional[bool],
        quantize: Optional[str],
        trust_remote_code: bool,
    ):
        from text_generation_server.cli import download_weights
        from text_generation_server.models import get_model

        lock_path = os.path.expanduser(f"~/{model_id.replace('/', '--')}.lock")
        print("downloading weights")
        with FileLock(lock_path):
            download_weights(model_id=model_id, revision=revision)
        print("weights downloaded")
        with patch(
            "text_generation_server.utils.dist.initialize_torch_distributed",
            initialize_torch_distributed,
        ), patch(
            "text_generation_server.models.gpt_neox.initialize_torch_distributed",
            initialize_torch_distributed,
        ):  # ExitStack() as stack:
            # mgrs = [patch(f"{module}.initialize_torch_distributed") for module in sys.modules if module.startswith("text_generation_server.models")]
            # indices_to_remove = []
            # for i, mgr in enumerate(mgrs):
            #     try:
            #         stack.enter_context(mgr)
            #     except AttributeError:
            #         indices_to_remove.append(i)
            # for i in reversed(indices_to_remove):
            #     mgrs.pop(i)
            print("get model")
            super().__init__(
                lambda: get_model(
                    model_id=model_id,
                    revision=revision,
                    sharded=int(os.getenv("WORLD_SIZE", "1")) > 1
                    if sharded is None
                    else sharded,
                    quantize=quantize,
                    trust_remote_code=trust_remote_code,
                )
            )
