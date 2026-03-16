"""Tests for the Flask web application."""

import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PIL import Image

import app as app_module
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app_module._pattern_store.clear()
    app_module._conversations.clear()
    with app.test_client() as client:
        yield client


def _make_png_bytes(width: int = 50, height: int = 50, color=(255, 100, 50)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestIndexRoute:
    def test_index_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "DMC" in resp.data.decode()

    def test_index_shows_upload_form(self, client):
        resp = client.get("/")
        body = resp.data.decode()
        assert "enctype=\"multipart/form-data\"" in body
        assert "/upload" in body


class TestUploadRoute:
    def test_upload_no_file(self, client):
        resp = client.post("/upload", data={})
        assert resp.status_code == 200
        assert "Файл не выбран" in resp.data.decode()

    def test_upload_invalid_extension(self, client):
        resp = client.post(
            "/upload",
            data={"image": (io.BytesIO(b"not an image"), "test.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert "Недопустимый формат" in resp.data.decode()

    def test_upload_valid_image_redirects(self, client):
        png_data = _make_png_bytes()
        resp = client.post(
            "/upload",
            data={
                "image": (io.BytesIO(png_data), "test.png"),
                "width": "20",
                "height": "0",
                "max_colors": "10",
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "/pattern/" in resp.headers["Location"]

    def test_upload_creates_pattern(self, client):
        png_data = _make_png_bytes()
        resp = client.post(
            "/upload",
            data={
                "image": (io.BytesIO(png_data), "test.png"),
                "width": "20",
                "height": "10",
                "max_colors": "5",
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        body = resp.data.decode()
        assert "Схема" in body
        assert "крестиков" in body


class TestPatternRoute:
    def _upload_and_get_id(self, client) -> str:
        png_data = _make_png_bytes()
        resp = client.post(
            "/upload",
            data={
                "image": (io.BytesIO(png_data), "test.png"),
                "width": "15",
                "height": "8",
                "max_colors": "5",
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        location = resp.headers["Location"]
        return location.split("/pattern/")[1]

    def test_view_pattern_ok(self, client):
        pid = self._upload_and_get_id(client)
        resp = client.get(f"/pattern/{pid}")
        assert resp.status_code == 200
        assert "DMC" in resp.data.decode()

    def test_view_pattern_not_found(self, client):
        resp = client.get("/pattern/notexists")
        assert resp.status_code == 404

    def test_pattern_html_contains_table(self, client):
        pid = self._upload_and_get_id(client)
        resp = client.get(f"/pattern/{pid}")
        body = resp.data.decode()
        assert "<table" in body
        assert "Легенда цветов" in body


class TestApiPatternRoute:
    def _upload_and_get_id(self, client) -> str:
        png_data = _make_png_bytes()
        resp = client.post(
            "/upload",
            data={
                "image": (io.BytesIO(png_data), "test.png"),
                "width": "10",
                "height": "5",
                "max_colors": "5",
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        return resp.headers["Location"].split("/pattern/")[1]

    def test_api_returns_json(self, client):
        import json
        pid = self._upload_and_get_id(client)
        resp = client.get(f"/api/pattern/{pid}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["width"] == 10
        assert data["height"] == 5
        assert "grid" in data
        assert "color_legend" in data

    def test_api_not_found(self, client):
        resp = client.get("/api/pattern/notexists")
        assert resp.status_code == 404


class TestChatRoute:
    def test_chat_get(self, client):
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert "Чат с Claude AI" in resp.data.decode()

    def test_chat_post_no_key(self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        resp = client.post(
            "/chat",
            data={"session_id": "test123", "message": "Hello"},
        )
        assert resp.status_code == 200
        body = resp.data.decode()
        assert "ANTHROPIC_API_KEY" in body

    def test_chat_clear(self, client):
        app_module._conversations["sess1"] = [{"role": "user", "content": "hi"}]
        resp = client.post("/chat/clear", data={"session_id": "sess1"})
        assert resp.status_code == 302
        assert "sess1" not in app_module._conversations


class TestSuggestRoute:
    def test_suggest_get(self, client):
        resp = client.get("/suggest")
        assert resp.status_code == 200
        assert "Советы" in resp.data.decode()

    def test_suggest_post_no_key(self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        resp = client.post("/suggest", data={"description": "хочу вышить цветок"})
        assert resp.status_code == 200
        body = resp.data.decode()
        assert "ANTHROPIC_API_KEY" in body


class TestAllowedFile:
    def test_allowed(self):
        from app import allowed_file
        for ext in ("jpg", "jpeg", "png", "gif", "bmp", "webp"):
            assert allowed_file(f"test.{ext}")

    def test_not_allowed(self):
        from app import allowed_file
        for ext in ("txt", "pdf", "exe", "py"):
            assert not allowed_file(f"test.{ext}")
