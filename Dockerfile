FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import db; db.init_db()"

ENV PORT=5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 2 app:app"]