from fastapi import FastAPI, HTTPException
from app.model import ToxicityClassifier
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Russian Toxicity Classifier API",
    description="API для классификации токсичности русских текстов",
    version="1.0.0"
)

# Инициализация модели при запуске
classifier = ToxicityClassifier()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check requested")
    return HealthResponse(status="OK")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single_text(request: PredictionRequest):
    try:
        logger.info(f"Predicting toxicity for text: {request.text[:100]}...")
        result = classifier.predict(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch_texts(request: BatchPredictionRequest):
    try:
        logger.info(f"Predicting toxicity for {len(request.texts)} texts")
        results = classifier.predict_batch(request.texts)
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    try:
        model_info = classifier.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
