import time
from collections import deque
from dataclasses import dataclass
from threading import Condition, RLock
from typing import Optional

from text_generation_server.pb.generate_pb2 import (
    Request as GenerationRequest,
)

from hf_shim.tgi.tokenstream import TokenStream


@dataclass
class InferenceRequest:
    id: int
    request: GenerationRequest
    request_input_length: int
    output_stream: TokenStream
    submit_time_ns: int

    @classmethod
    def from_request(cls, request: GenerationRequest, request_input_length: int):
        return cls(
            id=request.id,
            request=request,
            request_input_length=request_input_length,
            output_stream=TokenStream(),
            submit_time_ns=int(time.time()),
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.request_input_length + self.request.stopping_parameters.max_new_tokens
        )

    @property
    def input_length(self) -> int:
        return self.request_input_length + self.output_stream.num_tokens()

    @property
    def gen_length(self) -> int:
        return max(
            0,
            self.request.stopping_parameters.max_new_tokens
            - self.output_stream.num_tokens,
        )


class RequestQueue:
    def __init__(self):
        self._queue = deque()
        self._lock = RLock()
        self._cv = Condition(self._lock)

    def push(self, request: InferenceRequest) -> bool:
        with self._cv:
            self._queue.append(request)
            self._cv.notify_all()
            return True

    def peek(self) -> Optional[InferenceRequest]:
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue[0]

    def pop(self) -> Optional[InferenceRequest]:
        with self._lock:
            while len(self._queue) == 0:
                return None
            return self._queue.popleft()

    def wait(self, timeout=None):
        start = time.time()
        with self._cv:
            while len(self._queue) == 0:
                self._cv.wait(timeout)
                if timeout is not None and time.time() - start >= timeout:
                    return

    def reverse_push(self, request: InferenceRequest) -> None:
        with self._cv:
            self._queue.appendleft(request)
            self._cv.notify_all()

    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
