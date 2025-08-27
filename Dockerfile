# Using small official python image
FROM python:3.11-slim

# System libs that ML stacks often need (xgboost needs libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Avoiding buffering / ensure predictable output
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Workdir
WORKDIR /app

# Copy only dependency files first for better caching
COPY requirements.txt /app/

# Install deps (wheels will be used for numpy/pandas/sklearn on slim)
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy the rest of your app
COPY . /app

# Expose the port App Runner will hit
ENV PORT=8080
EXPOSE 8080

# Start with gunicorn (make sure 'gunicorn' is in requirements.txt)
# My Flask object is "app" inside app.py -> "app:app"
ENV PORT=8080
EXPOSE 8080

CMD gunicorn -w 1 -k gthread -b 0.0.0.0:$PORT app:app
