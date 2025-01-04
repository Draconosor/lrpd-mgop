from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Callable, Dict


@dataclass
class FunctionTimer:
    """Utility for timing function executions."""
    def __init__(self):
        self.execution_times: Dict[str, list] = defaultdict(list)
        self.total_executions: Dict[str, int] = defaultdict(int)
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            time_taken = perf_counter() - start
            
            func_name = func.__name__
            self.execution_times[func_name].append(time_taken)
            self.total_executions[func_name] += 1
            
            return result
        return wrapper
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for func_name, times in self.execution_times.items():
            stats[func_name] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_calls': self.total_executions[func_name],
                'total_exec_time': sum(times)
            }
        return stats