from typing import Callable, List, TypeVar, Iterable, Generic, Optional
from threading import Thread, RLock, Condition


T = TypeVar('T')
U = TypeVar('U')


class ParallelMap(Generic[T, U]):

    def __init__(
        self,
        func: Callable[[T], U],
        iterable: Iterable[T],
        num_threads: int,
        size: int = None,
    ) -> None:
        self.func = func
        self.iterator = iter(iterable)
        self.num_threads = num_threads
        self.size = size or num_threads
        self.queue = []
        self.lock = RLock()
        self.condition = Condition(self.lock)
        self.stop_signal = False
        self.threads: List[Thread] = []
        self.num_active_threads = 0
        self.last_exception: Optional[Exception] = None

    def __enter__(self) -> 'ParallelMap[T, U]':
        self.num_active_threads = self.num_threads
        for _ in range(self.num_threads):
            thread = Thread(target=self._produce)
            thread.start()
            self.threads.append(thread)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_signal = True
        with self.lock:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join()

    def __iter__(self) -> 'ParallelMap[T, U]':
        if self.num_active_threads == 0:
            raise RuntimeError("ParallelMap must be used as a context manager")
        return self

    def __next__(self) -> U:
        with self.lock:
            # Wait for an item if there are active threads
            while self.num_active_threads > 0 and not self.queue:
                self.condition.wait()

            # Return the item if available
            if self.queue:
                item = self.queue.pop(0)
                self.condition.notify_all()
                return item

            # Check if there was an exception
            if self.last_exception:
                raise self.last_exception
            raise StopIteration

    def _produce(self):
        while True:
            # Check for stop signal
            if self.stop_signal:
                break

            with self.lock:
                # Get the next input item (and check for end of input)
                try:
                    input_item = next(self.iterator)
                except StopIteration:
                    break

            # Process the input
            try:
                item = self.func(input_item)
            except Exception as e:  # pylint: disable=broad-except
                # On exception, signal stop and exit
                with self.lock:
                    self.stop_signal = True
                    self.last_exception = e
                break

            # Add the processed item to the queue
            with self.lock:
                # Wait while the queue is full or until stop signal
                while len(self.queue) >= self.size and not self.stop_signal:
                    self.condition.wait()

                # Add item to the queue
                self.queue.append(item)
                self.condition.notify_all()

        # Decrement the active thread count
        with self.lock:
            self.num_active_threads -= 1
            self.condition.notify_all()
