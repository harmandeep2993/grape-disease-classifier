FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & sleep 10 && python frontend/nicegui_app.py"]

# CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & sleep 10 && uv run python frontend/nicegui_app.py --server.port 7860 --server.address 0.0.0.0"]