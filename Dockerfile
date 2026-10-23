FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir --no-deps -e .
COPY web/ ./web/
# Generate the reproducible demonstration model during the image build.
RUN python -m digit_classifier.train
EXPOSE 8000
CMD ["uvicorn", "digit_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
