# tests/conftest.py
import os
import importlib
import pytest

@pytest.fixture(autouse=True)
def disable_plots(monkeypatch):
    """
    Force matplotlib to use a non-interactive backend and monkeypatch plt.show
    so tests do not open GUI windows. This fixture is autouse so it applies
    to all tests in the test session.
    """
    # Set MPL backend env var early so matplotlib picks it up on import
    os.environ.setdefault("MPLBACKEND", "Agg")

    # If matplotlib already imported, switch backend where possible:
    try:
        import matplotlib
        # matplotlib.use must be called before importing pyplot in some environments.
        # If pyplot already imported, we still monkeypatch show below.
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
    except Exception:
        pass

    # Now monkeypatch pyplot.show to a no-op if pyplot is imported or later imported
    def _noop_show(*args, **kwargs):
        return None

    # If pyplot already imported, patch show immediately
    try:
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", _noop_show)
    except Exception:
        # If pyplot not imported yet, patch importlib to patch when it gets imported
        orig_import = importlib.import_module

        def _import_and_patch(name, package=None):
            mod = orig_import(name, package=package)
            if name == "matplotlib.pyplot":
                try:
                    import matplotlib.pyplot as _plt  # noqa: F401
                    monkeypatch.setattr(_plt, "show", _noop_show)
                except Exception:
                    pass
            return mod

        monkeypatch.setattr(importlib, "import_module", _import_and_patch)

    # Also monkeypatch plt.pause if present (sometimes used for interactive loops)
    try:
        import matplotlib.pyplot as plt
        if hasattr(plt, "pause"):
            monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
    except Exception:
        pass

    # If code uses tkinter to create windows, monkeypatch Tk to be a dummy (optional)
    try:
        import tkinter as tk
        class _DummyTk:
            def __init__(self, *a, **k): pass
            def withdraw(self, *a, **k): pass
            def destroy(self, *a, **k): pass
        monkeypatch.setattr(tk, "Tk", _DummyTk)
    except Exception:
        pass

    yield
