# Triton образ с Python backend
FROM nvcr.io/nvidia/tritonserver:24.11-py3

# Зависимости
RUN pip install --no-cache-dir \
    transformers \
    torch \
    numpy \
    tritonclient[http]

# Копирование model_repository в контейнер
COPY model_repository /models

# Открытие портов для Triton
EXPOSE 8000 8001 8002

# Запуск Triton Server
CMD ["tritonserver", \
     "--model-repository=/models", \
     "--strict-model-config=false", \
     "--log-verbose=1", \
     "--model-control-mode=explicit", \
     "--load-model=toxicity_classifier"]