"""Smoke test: main.py must import cleanly on the minimum supported Python
version (3.10) -- none of the other tests import it, so a syntax feature
gated to a newer version (e.g. a backslash inside a nested f-string
expression, 3.12+ only) would otherwise go unnoticed until a user on 3.10/3.11
tried to run the CLI."""

import importlib


def test_main_module_imports():
    importlib.import_module("re_search_it.main")
