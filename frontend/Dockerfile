FROM python:3.13-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .

EXPOSE 7860
CMD ["python", "-m", "gradio", "app_gradio.py"]