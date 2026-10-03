"""Pytest configuration: tests never touch the network (CLAUDE.md section 9)."""
import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("network access in tests is not allowed (CLAUDE.md section 9)")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
