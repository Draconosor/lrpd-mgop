from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Callable, Dict, TypeVar, ParamSpec, Union

P = ParamSpec("P")
R = TypeVar("R")

@dataclass
class FunctionTimer:
    """Utility for timing function executions."""
    def __init__(self):
        self.execution_times: Dict[str, list] = defaultdict(list)
        self.total_executions: Dict[str, int] = defaultdict(int)
    
    def __call__(self, func: Callable[P,R]) -> Union[Callable[P,R], None]:
        """
        A decorator that measures the execution time of the decorated function and stores it.
        Args:
            func (Callable[P, R]): The function to be decorated.
        Returns:
            Union[Callable[P, R], None]: The wrapped function with execution time measurement.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            A decorator function that wraps another function to measure its execution time.
            Args:
                *args: Variable length argument list for the wrapped function.
                **kwargs: Arbitrary keyword arguments for the wrapped function.
            Returns:
                The result of the wrapped function.
            Side Effects:
                Records the execution time of the wrapped function in `self.execution_times`.
                Increments the total execution count of the wrapped function in `self.total_executions`.
            """
            start = perf_counter()
            result = func(*args, **kwargs)
            time_taken = perf_counter() - start
            
            func_name = func.__name__
            self.execution_times[func_name].append(time_taken)
            self.total_executions[func_name] += 1
            
            return result
        return wrapper
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate and return statistics for each tracked function.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary where each key is a function name and the value is another dictionary
            containing the following statistics:
                - 'avg_time': The average execution time of the function.
                - 'min_time': The minimum execution time of the function.
                - 'max_time': The maximum execution time of the function.
                - 'total_calls': The total number of times the function was called.
                - 'total_exec_time': The total execution time of the function.
        """
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
    
    def reset(self):
        """
        Reset the timer by clearing all recorded execution times and total executions.
        """
        self.execution_times.clear()
        self.total_executions.clear()