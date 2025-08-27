FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn
COPY . /app
ENV PORT=8080
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120
