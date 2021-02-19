"""profiler.py
Exposes a timing profiler decorator. Thanks to Antoine Toubhans.
https://medium.com/sicara/profile-surgical-time-tracking-python-db1e0a5c06b6
"""

import logging
import time
from functools import wraps

_total_time_call_stack = [0]

def timeprofiledecorator(log_fn):
    def _timeprofiledecorator(fn):
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
