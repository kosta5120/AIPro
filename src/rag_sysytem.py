"""Compatibility shim for the common typo ``rag_sysytem``.

Re-exports :mod:`rag_system` so imports and CLI usage match the documented
interface.
"""
from __future__ import annotations

from rag_system import answer, get_retriever

__all__ = ["answer", "get_retriever"]

if __name__ == "__main__":
    import os
    import runpy

    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "rag_system.py"),
        run_name="__main__",
    )
