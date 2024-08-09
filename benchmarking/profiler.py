import torch
import contextlib
import time
from collections import defaultdict


class Profiler:
    _instance = None

    def __new__(cls, should_profile: bool = False, benchmark: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.profiler = torch.profiler.profile(record_shapes=True, with_flops=True, profile_memory=True, with_stack=True, with_modules=True) if should_profile else None
            cls._instance.benchmark = benchmark
            cls._instance.benchmark_vals = defaultdict(list)
            cls._instance.function_stack = []

        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None
    
    @classmethod
    def profiling_decorator(cls, record_name: str = None, skip_profiling: bool = False, skip_benchmark: bool = False):
        def inner(func):
            def wrapper(*args, **kwargs):
                if not cls._instance or (skip_profiling and skip_benchmark):
                    return func(*args, **kwargs)
                cls._instance.function_stack.append(record_name or func.__name__)
                name = ".".join(cls._instance.function_stack)
                if cls._instance.profiler and not skip_profiling:
                    cls._instance.profiler.start()
                start_time = time.perf_counter()
                
                with torch.profiler.record_function(name):
                    result = func(*args, **kwargs)
                
                end_time = time.perf_counter()
                if cls._instance.benchmark and not skip_benchmark:
                    cls._instance.benchmark_vals[name].append(end_time - start_time)
                if cls._instance.profiler and not skip_profiling:
                    cls._instance.profiler.stop()
                cls._instance.function_stack.pop()
                return result
            return wrapper
        return inner
    
    @classmethod
    def step(cls):
        if cls._instance and cls._instance.profiler:
            cls._instance.profiler.step()
    
    @classmethod
    def get_benchmark_vals(cls):
        if cls._instance and cls._instance.benchmark:
            return {k: sum(v) / len(v) for k, v in cls._instance.benchmark_vals.items()}
        return None
    
    @classmethod
    def get_profiling_data(cls):
        if cls._instance and  cls._instance.profiler:
            return self.profiler.key_averages()
        return None
    
