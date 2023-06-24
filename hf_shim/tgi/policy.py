import asyncio
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

from hf_shim.tgi.queue import InferenceRequest


class RequestSelectionPolicy(ABC):
    @abstractmethod
    def select_new_requests(
        self,
        in_process_requests: List[InferenceRequest],
        queue: asyncio.Queue,
    ) -> List[InferenceRequest]:
        raise NotImplementedError

    @abstractmethod
    def request_finished(self, finished_request: InferenceRequest):
        raise NotImplementedError


Quota = namedtuple("Quota", ["min_num_requests", "token_budget"])


class QuotaBasedRequestSelectionPolicy(RequestSelectionPolicy):
    def __init__(
        self,
        max_batch_total_tokens: int = 32000,
        waiting_served_ratio: float = 1.2,
        max_waiting_tokens: int = 20,
    ):
        self.max_batch_total_tokens = max_batch_total_tokens
        self.waiting_served_ratio = waiting_served_ratio
        self.max_waiting_tokens = max_waiting_tokens
        self.waiting_tokens = 0
        self.oom_penalty = 1.0
        self.oomed_requests = set()

    def request_finished(self, finished_request: InferenceRequest):
        if finished_request.id in self.oomed_requests:
            self.oomed_requests.remove(finished_request.id)
        if len(self.oomed_requests) == 0:
            self.oom_penalty = 1

    def _calculate_budget(
        self,
        in_process: List[InferenceRequest],
        selected: List[InferenceRequest],
        candidate: InferenceRequest,
    ):
        max_input_length = candidate.input_length
        gen_length = candidate.gen_length
        for r in in_process:
            max_input_length = max(max_input_length, r.input_length)
            gen_length += r.gen_length
        for r in selected:
            max_input_length = max(max_input_length, r.input_length)
            gen_length += r.gen_length
        return gen_length + max_input_length * (len(selected) + 1 + len(in_process))

    def select_new_requests(
        self,
        in_process_requests: List[InferenceRequest],
        queue: asyncio.Queue,
        has_oom: bool = False,
    ) -> List[InferenceRequest]:
        if has_oom:
            self.oom_penalty = 0.7
            for r in in_process_requests:
                self.oomed_requests.add(r.id)

        min_num_requests, token_budget = self.calculate_quota(
            in_process_requests, has_oom=False
        )

        if min_num_requests and queue.qsize() < min_num_requests:
            return []

        hypothetical_results = []
        while len(hypothetical_results) < queue.qsize():
            request = queue._queue[0]
            if request.total_tokens >= token_budget:
                break
            hypothetical_results.append(request)
            token_budget -= request.total_tokens

        results = []
        if min_num_requests and len(hypothetical_results) < min_num_requests:
            results = []
        else:
            results = []
            for _ in hypothetical_results:
                print("getting from queue")
                results.append(queue.get_nowait())

        return results

    def calculate_quota(
        self, in_process_requests: List[InferenceRequest], has_oom: bool = False
    ) -> Quota:
        if not in_process_requests:
            return Quota(
                min_num_requests=None,
                token_budget=int(self.max_batch_total_tokens * self.oom_penalty),
            )

        batch_size = len(in_process_requests)

        # calculate minmal_new_requests to be served
        if self.waiting_tokens >= self.max_waiting_tokens:
            pass
        else:
            int(batch_size * self.waiting_served_ratio)

        return Quota(
            min_num_requests=None,
            token_budget=int(self.max_batch_total_tokens * self.oom_penalty),
        )
