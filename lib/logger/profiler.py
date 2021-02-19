"""profiler.py
Exposes a timing profiler decorator. Thanks to Antoine Toubhans.
https://medium.com/sicara/profile-surgical-time-tracking-python-db1e0a5c06b6
"""

import logging
import time
from functools import wraps

def _profiler_print_fn(data_dict):
    logging.debug(f'[Profiler] {data_dict["function_name"]} total: {data_dict["total_time"]:.3f} partial: {data_dict["partial_time"]:.3f}')
    return

_total_time_call_stack = [0]

def timeprofiledecorator(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        global _total_time_call_stack
        _total_time_call_stack.append(0)
        # Start timer
        start_time = time.time()
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed_time = time.time() - start_time
            inner_total_time = _total_time_call_stack.pop()
            partial_time = elapsed_time - inner_total_time
            _total_time_call_stack[-1] += elapsed_time
            # Log the result
            _profiler_print_fn({
                'function_name': fn.__name__,
                'total_time'   : elapsed_time,
                'partial_time' : partial_time
            })
        return result
    return wrapped_fn
