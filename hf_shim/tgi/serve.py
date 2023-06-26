"""This file contains objects that Serve users should interact with directly.

It Serve deployment classes that users can inherit from and decorate with
@serve.deployment.
"""

import random
import asyncio
from typing import List
from starlette.requests import Request
from starlette.responses import StreamingResponse

from transformers import AutoTokenizer

from hf_shim.tgi.params import SamplingParams
from hf_shim.tgi.policy import QuotaBasedRequestSelectionPolicy
from hf_shim.tgi.scheduler import InferenceScheduler, TransfomerTokenizer
from hf_shim.tgi.worker import TGIInferenceWorker


class TGIServer:
    def __init__(
        self,
        model_id: str,
        max_batch_total_tokens: int = 25000,
        waiting_served_ratio: float = 1.2,
        max_waiting_tokens: int = 20,
        max_length: int = 512,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        typical_p: float = 1.0,
        do_sample: bool = False,
        sampling_params_seed: int = 42,
        max_new_tokens: int = 64,
        stop_sequences: List[str] = [],
        ignore_eos_token: bool = False,
        watermark: bool = False,
    ):
        random.seed(0xCADE)

        request_selection_policy = QuotaBasedRequestSelectionPolicy(
            max_batch_total_tokens=max_batch_total_tokens,
            waiting_served_ratio=waiting_served_ratio,
            max_waiting_tokens=max_waiting_tokens,
        )

        self.params = SamplingParams(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            do_sample=do_sample,
            seed=sampling_params_seed,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
            ignore_eos_token=ignore_eos_token,
            watermark=watermark,
        )

        self.scheduler = InferenceScheduler(
            tokenizer=TransfomerTokenizer(
                pretrained_model_name_or_path=model_id, padding_side="left"
            ),
            inference_worker_loader=lambda: TGIInferenceWorker(
                model_id=model_id,
                revision=None,
                sharded=None,
                quantize=None,
                trust_remote_code=True,
            ),
            request_selection_policy=request_selection_policy,
            request_queue=asyncio.Queue(),
            inline=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.max_length = max_length

    async def __call__(self, request: Request):
        prompt = request.query_params["prompt"]
        result_stream = self.scheduler.process_request(
            prompt, params=self.params, max_length=self.max_length
        )
        return StreamingResponse(
            result_stream, status_code=200, media_type="text/plain"
        )
