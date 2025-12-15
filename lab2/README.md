docker build -t triton-toxicity .

docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-toxicity

$ curl -v localhost:8000/v2/health/ready

$ curl http://localhost:8002/metrics

python client.py