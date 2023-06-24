from dataclasses import dataclass
from typing import List


@dataclass
class SamplingParams:
    temperature: float
    top_k: int
    top_p: float
    typical_p: float
    do_sample: bool
    seed: int
    repetition_penalty: float
    watermark: bool

    max_new_tokens: int
    stop_sequences: List[str]
    ignore_eos_token: bool


__all__ = ["SamplingParams"]
