"""Register a minimal pkg_resources shim before face_recognition imports (Python 3.12+)."""
import importlib.resources
import sys
import types

try:
    import pkg_resources as _pr  # noqa: F401
except ImportError:
    _pr = None

if _pr is None or not hasattr(_pr, "resource_filename"):
    def resource_filename(package: str, resource: str) -> str:
        root = importlib.resources.files(package)
        p = root
        for part in resource.replace("\\", "/").split("/"):
            if part:
                p = p / part
        return str(p)

    m = types.ModuleType("pkg_resources")
    m.resource_filename = resource_filename
    sys.modules["pkg_resources"] = m
