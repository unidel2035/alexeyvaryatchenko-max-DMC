"""
Flask web server for DMC Embroidery Pattern Tool.

Endpoints:
  GET  /                  - Main page with upload form
  POST /upload            - Upload image and generate pattern
  GET  /pattern/<id>      - View saved pattern
  POST /ask               - Ask Claude AI about embroidery
  POST /suggest           - Get Claude AI suggestions for a design
  GET  /api/pattern/<id>  - JSON API for pattern data
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from image_processor import process_image
from claude_client import ask_claude, analyze_pattern_with_claude, suggest_pattern_improvements

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# In-memory pattern store {pattern_id: EmbroideryPattern}
_pattern_store: dict[str, object] = {}
# Per-session conversation histories {session_id: list[dict]}
_conversations: dict[str, list[dict]] = {}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

_BASE_STYLE = """
<style>
  * { box-sizing: border-box; }
  body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; color: #333; }
  header { background: #8B4513; color: white; padding: 16px 32px; }
  header h1 { margin: 0; font-size: 1.6rem; }
  .container { max-width: 1100px; margin: 32px auto; padding: 0 16px; }
  .card { background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px;
          box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
  .btn { background: #8B4513; color: white; border: none; padding: 10px 24px;
         border-radius: 4px; cursor: pointer; font-size: 1rem; text-decoration: none; display: inline-block; }
  .btn:hover { background: #A0522D; }
  .btn-secondary { background: #6c757d; }
  .btn-secondary:hover { background: #5a6268; }
  input[type=file], input[type=text], input[type=number], textarea, select {
    width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px;
    margin-top: 4px; font-size: 0.95rem; }
  textarea { height: 80px; resize: vertical; }
  label { font-weight: bold; margin-top: 12px; display: block; }
  .form-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .form-row > div { flex: 1; min-width: 160px; }
  .error { color: #c00; background: #fff0f0; border: 1px solid #fcc; padding: 12px;
           border-radius: 4px; margin-bottom: 16px; }
  .success { color: #060; background: #f0fff0; border: 1px solid #cfc; padding: 12px;
             border-radius: 4px; margin-bottom: 16px; }
  .chat-box { border: 1px solid #ddd; border-radius: 4px; padding: 12px;
              max-height: 400px; overflow-y: auto; background: #fafafa; margin-bottom: 8px; }
  .chat-msg { margin-bottom: 12px; }
  .chat-msg.user { text-align: right; }
  .chat-msg .bubble { display: inline-block; padding: 8px 14px; border-radius: 18px;
                       max-width: 80%; text-align: left; }
  .chat-msg.user .bubble { background: #8B4513; color: white; }
  .chat-msg.assistant .bubble { background: #e9ecef; color: #333; }
  nav a { color: white; text-decoration: none; margin-right: 16px; font-size: 0.95rem; }
  nav a:hover { text-decoration: underline; }
</style>
"""

_INDEX_TEMPLATE = (
    _BASE_STYLE
    + """
<header>
  <h1>🧵 DMC — Создание схем для вышивания</h1>
  <nav>
    <a href="/">Главная</a>
    <a href="/chat">Помощник AI</a>
    <a href="/suggest">Советы</a>
  </nav>
</header>
<div class="container">
  <div class="card">
    <h2>Загрузить изображение</h2>
    <p>Загрузите любое изображение, чтобы преобразовать его в схему для вышивания крестиком с подбором цветов DMC.</p>
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <form method="POST" action="/upload" enctype="multipart/form-data">
      <label>Изображение (JPEG, PNG, GIF, BMP, WebP)</label>
      <input type="file" name="image" accept="image/*" required>
      <div class="form-row" style="margin-top:12px;">
        <div>
          <label>Ширина схемы (крестики)</label>
          <input type="number" name="width" value="60" min="10" max="300">
        </div>
        <div>
          <label>Высота схемы (крестики, 0 = авто)</label>
          <input type="number" name="height" value="0" min="0" max="300">
        </div>
        <div>
          <label>Макс. кол-во цветов DMC</label>
          <input type="number" name="max_colors" value="25" min="2" max="50">
        </div>
      </div>
      <button type="submit" class="btn" style="margin-top:16px;">Создать схему</button>
    </form>
  </div>

  {% if patterns %}
  <div class="card">
    <h2>Сохранённые схемы</h2>
    <ul>
      {% for pid, info in patterns %}
      <li><a href="/pattern/{{ pid }}">{{ info }}</a></li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}
</div>
"""
)

_PATTERN_TEMPLATE = (
    _BASE_STYLE
    + """
<header>
  <h1>🧵 DMC — Схема вышивания</h1>
  <nav>
    <a href="/">Главная</a>
    <a href="/chat">Помощник AI</a>
    <a href="/suggest">Советы</a>
  </nav>
</header>
<div class="container">
  <div class="card">
    <h2>{{ title }}</h2>
    <div style="margin-bottom:12px;">
      <a href="/" class="btn btn-secondary">← Назад</a>
      <a href="/api/pattern/{{ pattern_id }}" class="btn btn-secondary" style="margin-left:8px;">Скачать JSON</a>
      <a href="/ask-pattern/{{ pattern_id }}" class="btn" style="margin-left:8px;">Спросить AI об этой схеме</a>
    </div>
    {{ pattern_html | safe }}
  </div>
</div>
"""
)

_CHAT_TEMPLATE = (
    _BASE_STYLE
    + """
<header>
  <h1>🧵 DMC — AI Помощник</h1>
  <nav>
    <a href="/">Главная</a>
    <a href="/chat">Помощник AI</a>
    <a href="/suggest">Советы</a>
  </nav>
</header>
<div class="container">
  <div class="card">
    <h2>Чат с Claude AI</h2>
    <p>Задавайте вопросы о вышивании, DMC нитях, выборе схем и технике.</p>
    {% if no_key %}
    <div class="error">
      <strong>ANTHROPIC_API_KEY не установлен.</strong>
      Для использования AI функций установите переменную окружения ANTHROPIC_API_KEY.
    </div>
    {% endif %}
    <div class="chat-box" id="chat-box">
      {% for msg in history %}
      <div class="chat-msg {{ msg.role }}">
        <div class="bubble">{{ msg.content }}</div>
      </div>
      {% endfor %}
    </div>
    <form method="POST" action="/chat">
      <input type="hidden" name="session_id" value="{{ session_id }}">
      <label>Ваш вопрос</label>
      <div style="display:flex;gap:8px;margin-top:4px;">
        <input type="text" name="message" placeholder="Например: какие цвета DMC подойдут для розы?" required style="flex:1;">
        <button type="submit" class="btn">Отправить</button>
      </div>
    </form>
    <form method="POST" action="/chat/clear" style="margin-top:8px;">
      <input type="hidden" name="session_id" value="{{ session_id }}">
      <button type="submit" class="btn btn-secondary">Очистить историю</button>
    </form>
  </div>
</div>
<script>
  var box = document.getElementById('chat-box');
  if (box) box.scrollTop = box.scrollHeight;
</script>
"""
)

_SUGGEST_TEMPLATE = (
    _BASE_STYLE
    + """
<header>
  <h1>🧵 DMC — Советы по дизайну</h1>
  <nav>
    <a href="/">Главная</a>
    <a href="/chat">Помощник AI</a>
    <a href="/suggest">Советы</a>
  </nav>
</header>
<div class="container">
  <div class="card">
    <h2>Получить совет по схеме</h2>
    <p>Опишите, что вы хотите вышить, и AI поможет выбрать размер, количество цветов и технику.</p>
    {% if no_key %}
    <div class="error">
      <strong>ANTHROPIC_API_KEY не установлен.</strong>
      Для использования AI функций установите переменную окружения ANTHROPIC_API_KEY.
    </div>
    {% endif %}
    <form method="POST" action="/suggest">
      <label>Описание вашей идеи</label>
      <textarea name="description" placeholder="Например: хочу вышить пейзаж с горами и закатом, средний уровень сложности" required></textarea>
      <button type="submit" class="btn" style="margin-top:12px;">Получить совет</button>
    </form>
    {% if suggestion %}
    <div class="card" style="margin-top:16px;background:#f9f5f1;">
      <h3>Рекомендации AI:</h3>
      <div style="white-space:pre-wrap;">{{ suggestion }}</div>
    </div>
    {% endif %}
  </div>
</div>
"""
)

_ASK_PATTERN_TEMPLATE = (
    _BASE_STYLE
    + """
<header>
  <h1>🧵 DMC — AI анализ схемы</h1>
  <nav>
    <a href="/">Главная</a>
    <a href="/chat">Помощник AI</a>
    <a href="/suggest">Советы</a>
  </nav>
</header>
<div class="container">
  <div class="card">
    <h2>Спросить AI о схеме</h2>
    {% if no_key %}
    <div class="error">
      <strong>ANTHROPIC_API_KEY не установлен.</strong>
      Для использования AI функций установите переменную окружения ANTHROPIC_API_KEY.
    </div>
    {% endif %}
    <form method="POST" action="/ask-pattern/{{ pattern_id }}">
      <label>Вопрос о схеме (необязательно)</label>
      <input type="text" name="question" placeholder="Например: сколько времени займёт вышивание?">
      <button type="submit" class="btn" style="margin-top:12px;">Получить анализ</button>
    </form>
    {% if analysis %}
    <div class="card" style="margin-top:16px;background:#f9f5f1;">
      <h3>Анализ AI:</h3>
      <div style="white-space:pre-wrap;">{{ analysis }}</div>
    </div>
    {% endif %}
    <a href="/pattern/{{ pattern_id }}" class="btn btn-secondary" style="margin-top:12px;">← Назад к схеме</a>
  </div>
</div>
"""
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    patterns = [
        (pid, f"Схема {p.width}×{p.height}, {len(p.color_legend)} цветов")
        for pid, p in _pattern_store.items()
    ]
    return render_template_string(_INDEX_TEMPLATE, error=None, patterns=patterns)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        patterns = list(_pattern_store.items())
        return render_template_string(_INDEX_TEMPLATE, error="Файл не выбран.", patterns=[])

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return render_template_string(
            _INDEX_TEMPLATE,
            error="Недопустимый формат файла. Поддерживаются: JPEG, PNG, GIF, BMP, WebP.",
            patterns=[],
        )

    try:
        width = int(request.form.get("width", 60))
        height_val = int(request.form.get("height", 0))
        max_colors = int(request.form.get("max_colors", 25))

        width = max(10, min(300, width))
        max_colors = max(2, min(50, max_colors))
        pattern_height = max(1, min(300, height_val)) if height_val > 0 else None

        image_data = file.read()
        pattern = process_image(
            image_data,
            pattern_width=width,
            pattern_height=pattern_height,
            max_colors=max_colors,
        )

        pattern_id = str(uuid.uuid4())[:8]
        _pattern_store[pattern_id] = pattern

        return redirect(url_for("view_pattern", pattern_id=pattern_id))

    except Exception as exc:
        return render_template_string(
            _INDEX_TEMPLATE, error=f"Ошибка обработки изображения: {exc}", patterns=[]
        )


@app.route("/pattern/<pattern_id>")
def view_pattern(pattern_id: str):
    pattern = _pattern_store.get(pattern_id)
    if pattern is None:
        return render_template_string(
            _INDEX_TEMPLATE, error=f"Схема '{pattern_id}' не найдена.", patterns=[]
        ), 404

    pattern_html = pattern.to_html_table()
    title = f"Схема {pattern.width}×{pattern.height} крестиков, {len(pattern.color_legend)} цветов DMC"
    return render_template_string(
        _PATTERN_TEMPLATE,
        pattern_html=pattern_html,
        pattern_id=pattern_id,
        title=title,
    )


@app.route("/api/pattern/<pattern_id>")
def api_pattern(pattern_id: str):
    pattern = _pattern_store.get(pattern_id)
    if pattern is None:
        return jsonify({"error": "Pattern not found"}), 404
    return app.response_class(
        response=pattern.to_json(),
        status=200,
        mimetype="application/json",
    )


@app.route("/chat", methods=["GET", "POST"])
def chat():
    no_key = not os.environ.get("ANTHROPIC_API_KEY")

    if request.method == "GET":
        session_id = request.args.get("session_id", str(uuid.uuid4())[:8])
        history = _conversations.get(session_id, [])
        return render_template_string(
            _CHAT_TEMPLATE, history=history, session_id=session_id, no_key=no_key
        )

    session_id = request.form.get("session_id", str(uuid.uuid4())[:8])
    message = request.form.get("message", "").strip()
    history = _conversations.get(session_id, [])

    if message and not no_key:
        try:
            response_text, updated_history = ask_claude(message, history)
            _conversations[session_id] = updated_history
            history = updated_history
        except Exception as exc:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Ошибка: {exc}"},
            ]
            _conversations[session_id] = history

    return render_template_string(
        _CHAT_TEMPLATE, history=history, session_id=session_id, no_key=no_key
    )


@app.route("/chat/clear", methods=["POST"])
def chat_clear():
    session_id = request.form.get("session_id", "")
    _conversations.pop(session_id, None)
    return redirect(url_for("chat", session_id=session_id))


@app.route("/suggest", methods=["GET", "POST"])
def suggest():
    no_key = not os.environ.get("ANTHROPIC_API_KEY")
    suggestion = None

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        if description and not no_key:
            try:
                suggestion = suggest_pattern_improvements(description)
            except Exception as exc:
                suggestion = f"Ошибка: {exc}"
        elif description and no_key:
            suggestion = "ANTHROPIC_API_KEY не установлен — AI функции недоступны."

    return render_template_string(
        _SUGGEST_TEMPLATE, suggestion=suggestion, no_key=no_key
    )


@app.route("/ask-pattern/<pattern_id>", methods=["GET", "POST"])
def ask_pattern(pattern_id: str):
    no_key = not os.environ.get("ANTHROPIC_API_KEY")
    analysis = None

    pattern = _pattern_store.get(pattern_id)
    if pattern is None:
        return render_template_string(
            _INDEX_TEMPLATE, error=f"Схема '{pattern_id}' не найдена.", patterns=[]
        ), 404

    if request.method == "POST" and not no_key:
        question = request.form.get("question", "").strip()
        try:
            analysis = analyze_pattern_with_claude(pattern.to_json(), question)
        except Exception as exc:
            analysis = f"Ошибка: {exc}"
    elif request.method == "POST" and no_key:
        analysis = "ANTHROPIC_API_KEY не установлен — AI функции недоступны."

    return render_template_string(
        _ASK_PATTERN_TEMPLATE,
        pattern_id=pattern_id,
        analysis=analysis,
        no_key=no_key,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting DMC Embroidery Pattern Tool on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
