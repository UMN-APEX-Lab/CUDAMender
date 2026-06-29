"""Abstract backend interface and resolver."""

from abc import ABC, abstractmethod

from .task_schema import TaskConfig
from .result import PerfMetrics


class Backend(ABC):
    """Base class for task execution backends."""

    @abstractmethod
    def prepare(self, config: TaskConfig, solution_path: str) -> None:
        """Prepare the task (copy solution, set up environment)."""
        ...

    @abstractmethod
    def build(self, config: TaskConfig, solution_path: str) -> tuple[bool, str]:
        """Compile the solution. Returns (compiled, error_msg)."""
        ...

    @abstractmethod
    def run_correctness(self, config: TaskConfig) -> tuple[bool, dict, str]:
        """Check correctness. Returns (correct, detail_dict, error_msg)."""
        ...

    @abstractmethod
    def run_performance(self, config: TaskConfig) -> PerfMetrics:
        """Measure performance. Returns PerfMetrics."""
        ...

    @abstractmethod
    def cleanup(self, config: TaskConfig) -> None:
        """Clean up build artifacts."""
        ...


def resolve_backend(backend_name: str) -> Backend:
    """Resolve a backend name to a Backend instance.

    Only the Class 1 (Make-driven) and Class 2 (def.py / pybind11) backends
    are vendored in this runtime; application-level and frontier backends
    from the original benchmark are intentionally omitted.
    """
    if backend_name == "class1_make":
        from .backend_class1 import Class1MakeBackend
        return Class1MakeBackend()
    elif backend_name == "class2_defpy":
        from .backend_class2 import Class2DefpyBackend
        return Class2DefpyBackend()
    else:
        raise ValueError(f"Unknown or unsupported backend: {backend_name}")
