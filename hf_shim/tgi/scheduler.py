import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, List, Optional, Tuple

from ray._private.utils import run_background_task
from text_generation_server.pb.generate_pb2 import (
    NextTokenChooserParameters,
    StoppingCriteriaParameters,
)
from text_generation_server.pb.generate_pb2 import (
    Request as GenerationRequest,
)
from transformers import AutoTokenizer

from hf_shim.tgi.policy import QuotaBasedRequestSelectionPolicy
from hf_shim.tgi.queue import InferenceRequest
from hf_shim.tgi.tokenstream import TokenStream
from hf_shim.tgi.params import SamplingParams
from hf_shim.tgi.context import get_request_id

if TYPE_CHECKING:
    from text_generation_server.models.types import (
        Generation,
    )

logger = logging.getLogger(__name__)


class Tokenizer(ABC):
    @abstractmethod
    def get_input_length(self, input_text: str, max_length: int) -> int:
        raise NotImplementedError("")


class NaiveTokenizer(Tokenizer):
    def get_input_length(self, input_text: str, max_length: int) -> int:
        return min(input_text.count(" ") + 1, max_length)

    # TODO: add model specific tokenizer


class TransfomerTokenizer(Tokenizer):
    def __init__(self, pad_token_id=50256, *args, **kwargs):
        self._tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        self._tokenizer.pad_token_id = pad_token_id

    def get_input_length(self, input_text: str, max_length: int) -> int:
        return self._tokenizer(
            text=input_text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"].shape[1]


@dataclass
class Stats:
    num_requests_processed: int = 0
    num_requests_failed: int = 0
    num_requests_pending: int = 0
    num_active_requests: int = 0
    num_finished_requests: int = 0
    num_tokens_generated: int = 0
    num_input_tokens: int = 0
    num_iterations: int = 0
    last_report_time: float = 0.0
    start_time: float = 0.0

    def report_stats(self):
        if time.time() - self.last_report_time < 1:
            return False
        self.last_report_time = time.time()
        print(f"scheduler stats: {self}")
        elapsed = self.last_report_time - self.start_time
        token_s = (self.num_input_tokens + self.num_tokens_generated) / elapsed
        print(f"elapsed: {elapsed}, generated_tokens/s: {token_s}")
        return True

    def request_selected(self, requests: List[InferenceRequest]):
        self.num_active_requests += len(requests)
        self.num_requests_processed += len(requests)
        self.num_input_tokens += sum([r.input_length for r in requests])

    def request_finished(self):
        self.num_active_requests -= 1
        self.num_finished_requests += 1

    def request_failed(self):
        self.num_active_requests -= 1
        self.num_requests_failed += 1

    def token_generated(self, num):
        self.num_tokens_generated += num

    def iteration_finished(self):
        self.num_iterations += 1

    def set_num_requests_pending(self, num):
        self.num_requests_pending = num

    def start(self):
        self.start_time = time.time()


class InferenceScheduler:
    def __init__(
        self,
        tokenizer: Tokenizer,
        inference_worker_loader,
        request_selection_policy: QuotaBasedRequestSelectionPolicy,  # RequestSelectionPolicy,
        # request_queue: RequestQueue,
        request_queue: asyncio.Queue,
        inline: bool = False,
    ):
        self._tokenizer = tokenizer
        self._request_selection_policy = request_selection_policy
        self._inference_worker_loader = inference_worker_loader
        self._request_queue = request_queue
        self._queue_put_event = asyncio.Event()
        self._lock = Lock()
        self._stop = False
        self._stats = Stats()
        self._has_oom = False
        print("InferenceScheduler init")
        if not inline:
            # self._thread = Thread(target=self._run_scheduling_loop)
            # self._thread.start()
            # self.scheduling_loop_task = asyncio.get_event_loop().create_task(self._run_scheduling_loop())
            self.scheduling_loop_task = run_background_task(self._run_scheduling_loop())

    def stop(self):
        with self._lock:
            self._stop = True
        # self._thread.join()

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stop

    def process_request(
        self, input_text: str, params: SamplingParams, max_length: int = 1024
    ) -> TokenStream:
        params.max_new_tokens = max(1, params.max_new_tokens)
        request = GenerationRequest(
            id=get_request_id(),
            inputs=input_text,
            truncate=max_length,
            prefill_logprobs=False,
            parameters=NextTokenChooserParameters(
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                typical_p=params.typical_p,
                do_sample=params.do_sample,
                seed=params.seed,
                repetition_penalty=params.repetition_penalty,
                watermark=params.watermark,
            ),
            stopping_parameters=StoppingCriteriaParameters(
                max_new_tokens=params.max_new_tokens,
                stop_sequences=params.stop_sequences,
                ignore_eos_token=params.ignore_eos_token,
            ),
        )
        return self._add_request(request)

    def _add_request(self, request: GenerationRequest) -> TokenStream:
        pending_request = InferenceRequest.from_request(
            request,
            request_input_length=self._tokenizer.get_input_length(
                request.inputs, request.truncate
            ),
        )
        # self._request_queue.push(pending_request)
        self._request_queue.put_nowait(pending_request)
        self._queue_put_event.set()
        print("request added to queue")
        return pending_request.output_stream

    async def _run_scheduling_loop(self):
        """Schedule requests to be processed by the inference worker."""
        try:
            # start work the in the scheduling loop to avoid GPU memory leak.
            print("model loading...")
            self._inference_worker = self._inference_worker_loader()
            print("model loaded")
            self._stats.start()

            # The main schedule loop:
            #
            # 0. start with empty in-process requests.
            #
            # 1. select new requests to process, based
            # on the current in-process requests. send them to the inference worker.
            #
            # 2. for both new and in-process requests, combine them
            # and generate the next token. filter out finished requests.
            #
            # 3. goto step 1.
            batch_id = None
            in_process_requests = []
            await asyncio.sleep(0.000001)
            while not self.is_stopped():
                # select new requests to process.
                new_requests = await self._select_new_requests(in_process_requests)
                new_batch_id, new_unfinished_requests = self._process_new_requests(
                    new_requests
                )

                # combine new batch with existing batch to generate next token.
                batch_id, in_process_requests = self._generate_next_token(
                    [batch_id, new_batch_id],
                    in_process_requests + new_unfinished_requests,
                )

                self._stats.iteration_finished()
                self._report_stats()
                await asyncio.sleep(0.000001)
        except Exception:
            traceback.print_exc()
        finally:
            await asyncio.sleep(0.000001)

    def _report_stats(self):
        if self._stats.report_stats():
            self._stats.set_num_requests_pending(self._request_queue.qsize())
            self._inference_worker.report_stats()

    async def _select_new_requests(
        self,
        in_process_requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        while (
            len(in_process_requests) == 0
            and self._request_queue.empty()
            and not self.is_stopped()
        ):
            # if there is no in-process requests and no new requests in the queue,
            # wait for new requests to arrive in the queue.
            # self._request_queue.wait(1)

            await self._queue_put_event.wait()
            self._queue_put_event.clear()

        requests = self._request_selection_policy.select_new_requests(
            in_process_requests, self._request_queue
        )
        self._has_oom = False
        self._stats.request_selected(requests)
        return requests

    def _process_new_requests(
        self, requests: List[InferenceRequest]
    ) -> Tuple[int, List[InferenceRequest]]:
        if len(requests) == 0:
            return None, []
        generations, batch_id = self._inference_worker.process_new_batch(
            [r.request for r in requests]
        )
        requests, need_filter = self._process_generation_result(generations, requests)

        if need_filter and batch_id:
            batch_id = self._inference_worker.filter_requests(
                batch_id, [r.id for r in requests]
            )
        return batch_id, requests

    def _generate_next_token(
        self, batch_ids: List[int], requests: List[InferenceRequest]
    ) -> Tuple[Optional[int], List[InferenceRequest]]:
        generations, batch_id = self._inference_worker.generate_next_token(
            batch_ids,
        )

        # handle ooms
        if generations is None:
            return self._handle_ooms(batch_id, requests)

        requests, need_filter = self._process_generation_result(generations, requests)

        if batch_id is not None:
            if need_filter:
                batch_id = self._inference_worker.filter_requests(
                    batch_id, [r.id for r in requests]
                )
        else:
            assert len(requests) == 0, "expect no requests left"
        return batch_id, requests

    def _process_generation_result(
        self, generations: List["Generation"], requests: List[InferenceRequest]
    ) -> Tuple[List[InferenceRequest], bool]:
        some_request_finished = False
        unfinished_requests = []
        self._stats.token_generated(len(generations))
        for i, generation in enumerate(generations):
            assert (
                requests[i].id == generation.request_id
            ), f"expect request id {requests[i].id} but got {generation.request_id}"
            print(
                f"_process_generation_result: {generation.token_text} generated_text: {generation.generated_text}"
            )
            requests[i].output_stream.put(generation.token_text)
            if generation.generated_text is not None:
                self._stats.request_finished()
                requests[i].output_stream.end(generation.generated_text.text)
                some_request_finished = True
                self._request_selection_policy.request_finished(requests[i])
            else:
                unfinished_requests.append(requests[i])
        return unfinished_requests, some_request_finished

    def _handle_recoverable_ooms(self, batch_id, requests: List[InferenceRequest]):
        # pop last request to reduce memory overhead.
        assert requests
        failed_request = requests.pop()
        self._request_queue.reverse_push(failed_request)
        self._stats.request_failed()
        batch_id = self._inference_worker.filter_requests(
            batch_id, [r.id for r in requests]
        )
        self._has_oom = True
        return batch_id, requests

    def _handle_ooms(self, batch_id, requests: List[InferenceRequest]):
        if batch_id:
            return self._handle_recoverable_ooms(batch_id, requests)

        # oom is not recoverable
        while requests:
            failed_request = requests.pop()
            self._request_queue.reverse_push(failed_request)
            self._stats.request_failed()
        self._has_oom = True
        return None, []
