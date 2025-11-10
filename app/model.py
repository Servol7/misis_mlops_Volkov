import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ToxicityClassifier:
    def __init__(self, model_name: str = "s-nlp/russian_toxicity_classifier"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model {model_name} loaded successfully")

    def predict(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {
                "prediction": "non-toxic",
                "toxicity_score": 0.0,
                "is_toxic": False
            }
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            toxicity_score = probabilities[0][1].item()  # Второй класс - токсичный
        
        is_toxic = toxicity_score > 0.5
        prediction = "toxic" if is_toxic else "non-toxic"
        
        return {
            "prediction": prediction,
            "toxicity_score": round(toxicity_score, 4),
            "is_toxic": is_toxic
        }

    def predict_batch(self, texts: list) -> list:
        if not texts:
            return []
        
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return [{
                "text": text,
                "toxicity_score": 0.0,
                "is_toxic": False
            } for text in texts]
        
        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            toxicity_scores = probabilities[:, 1].cpu().numpy()  # Второй класс - токсичный
        
        results = []
        score_idx = 0
        
        for text in texts:
            if text.strip():
                toxicity_score = round(float(toxicity_scores[score_idx]), 4)
                is_toxic = toxicity_score > 0.5
                score_idx += 1
            else:
                toxicity_score = 0.0
                is_toxic = False
            
            results.append({
                "text": text,
                "toxicity_score": toxicity_score,
                "is_toxic": is_toxic
            })
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "max_length": self.tokenizer.model_max_length,
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size
        }
