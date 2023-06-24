import asyncio


class _EndOfStream:
    pass


EOS = _EndOfStream()


class TokenStream:
    """A stream of tokens that can be iterated over asynchronously."""

    def __init__(self, loop=None):
        self._queue = asyncio.Queue()
        self._loop = loop or asyncio.get_event_loop()
        self._num_tokens = 0
        self._generated_text = None

    def end(self, generated_text=None):
        self._generated_text = generated_text
        self._queue.put_nowait(EOS)

    def put(self, item):
        self._queue.put_nowait(item)
        self._num_tokens += 1

    def num_tokens(self):
        return self._num_tokens

    def __aiter__(self):
        return self

    async def __anext__(self):
        result = await self._queue.get()
        if result == EOS:
            raise StopAsyncIteration
        return result
