# Сборка образа
docker build -t triton-toxicity .

# Запуск сервера
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-toxicity

# Запуск клиента
python client/client.py