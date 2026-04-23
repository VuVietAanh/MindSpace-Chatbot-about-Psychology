# ============================================================
# Dockerfile — Mental Health Chatbot
# ============================================================

FROM python:3.11-slim

# Tránh hỏi interactive khi cài package
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Thư mục làm việc
WORKDIR /app

# Copy requirements trước (cache layer)
COPY requirements.txt .

# Cài Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ project
COPY . .

# Tạo thư mục cần thiết
RUN mkdir -p data/knowledge_base static/backgrounds static/chat_backgrounds

# Port FastAPI
EXPOSE 8001

# Chạy app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
