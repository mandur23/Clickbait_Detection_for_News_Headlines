from __future__ import annotations

from pathlib import Path

import pytest
from flask.testing import FlaskClient
from server_fakes import FakePredictor

from server.app import create_app
from server.history import HistoryStore


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def history(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "history.db")


@pytest.fixture
def client(predictor: FakePredictor) -> FlaskClient:
    return create_app(predictor).test_client()
