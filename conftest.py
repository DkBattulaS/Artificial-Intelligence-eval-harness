"""
Root conftest.py — shared pytest fixtures and configuration.
"""
import os
import sys

# Ensure the project root is on sys.path so `app.*` imports resolve
# without installing the package.
sys.path.insert(0, os.path.dirname(__file__))
