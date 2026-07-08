FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY protor/ protor/

RUN pip install --no-cache-dir .

CMD ["python", "-m", "protor"]
