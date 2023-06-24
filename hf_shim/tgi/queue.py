import time
from dataclasses import dataclass

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
