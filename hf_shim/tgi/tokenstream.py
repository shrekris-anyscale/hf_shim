import asyncio


class _Event_ts(asyncio.Event):
    """An asyncio.Event that can be used across threads."""

    def __init__(self):
        super().__init__()
        if not self._loop:
            self._loop = asyncio.get_event_loop()

    # TODO: clear() method
    def set(self):
        self._loop.call_soon_threadsafe(super().set)


# class TokenStream:
#     def __init__(self):
#         self._lock = RLock()
#         self._cv = Condition(self._lock)
#         self._num_tokens = 0
#         self._end = False
#         self._data = []
#         self._more_entry_event = _Event_ts()
#         self._finish_event = _Event_ts()
#         self._generated_text = None

#     def end(self, generated_text=None):
#         with self._lock:
#             self._end = True
#             self._generated_text = generated_text
#             self._cv.notify_all()

#         self._finish_event.set()
#         self._more_entry_event.set()

#     def put(self, item):
#         with self._lock:
#             self._data.append(item)
#             self._num_tokens += 1
#         self._more_entry_event.set()

#     def num_tokens(self):
#         with self._lock:
#             return self._num_tokens

#     def generated_text(self):
#         with self._lock:
#             return self._generated_text

#     def finished(self):
#         with self._lock:
#             return self._end

#     async def await_until_finished(self):
#         await self._finish_event.wait()
#         return

#     def wait_until_finished(self, timeout=None):
#         start = time.time()
#         with self._cv:
#             while not self._end:
#                 self._cv.wait(timeout)
#                 if timeout is not None and time.time() - start >= timeout:
#                     return

#     def __aiter__(self):
#         return self

#     async def __anext__(self):
#         while True:
#             with self._lock:
#                 if len(self._data) > 0:
#                     return self._data.pop(0)
#                 if self.finished():
#                     raise StopAsyncIteration
#             await self._more_entry_event.wait()
#             self._more_entry_event.clear()


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

    # async def end(self):
    # await asyncio.wrap_future(
    #     asyncio.run_coroutine_threadsafe(self._queue.put(EOS), self._loop)
    # )

    def end(self, generated_text=None):
        self._generated_text = generated_text
        self._queue.put_nowait(EOS)

    # async def put(self, item):
    def put(self, item):
        self._queue.put_nowait(item)
        self._num_tokens += 1
        # await asyncio.wrap_future(
        #     asyncio.run_coroutine_threadsafe(self._queue.put(item), self._loop)
        # )

    def num_tokens(self):
        return self._num_tokens

    def __aiter__(self):
        return self

    async def __anext__(self):
        print("TokenStream __anext__")
        result = await self._queue.get()
        print("TokenStream __anext__ has result")
        if result == EOS:
            print("raise StopAsyncIteration")
            raise StopAsyncIteration
        return result
