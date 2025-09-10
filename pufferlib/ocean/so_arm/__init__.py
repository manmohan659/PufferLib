import importlib as _importlib

__all__ = ["SoArm", "binding"]

def __getattr__(name):
    if name == "SoArm":
        from .so_arm import SoArm  # lazy to avoid circular import during package init
        return SoArm
    if name == "binding":
        # Load compiled extension lazily
        return _importlib.import_module(__name__ + ".binding")
    raise AttributeError(name)


