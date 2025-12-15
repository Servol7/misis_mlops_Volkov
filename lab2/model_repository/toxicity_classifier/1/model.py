import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch


class TritonPythonModel:
    def initialize(self, args):
        self.model_dir = args['model_repository']
        self.model_version = args['model_version']
        
        # Загружаем модель и токенизатор
        model_name = "s-nlp/russian_toxicity_classifier"
        
        try:
            # Инициализируем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Инициализируем модель
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Устанавливаем модель в режим оценки
            self.model.eval()
            
            print(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def execute(self, requests):
        """Execute inference for batch of requests"""
        responses = []
        
        for request in requests:
            # Получаем входные данные
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            texts = in_tensor.as_numpy().tolist()
            
            # Декодируем тексты (байты в строки)
            texts = [t.decode('utf-8') for t in texts]
            
            # Токенизируем тексты
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Выполняем инференс без вычисления градиентов
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Применяем softmax для получения вероятностей
            probabilities = F.softmax(logits, dim=-1)
            
            # Получаем предсказания (argmax)
            predictions = torch.argmax(logits, dim=-1)
            
            # Получаем скор токсичности (вероятность класса 1 - токсичный)
            toxicity_scores = probabilities[:, 1]
            
            # Создаем выходные тензоры
            logits_tensor = pb_utils.Tensor("LOGITS", logits.numpy().astype(np.float32))
            probs_tensor = pb_utils.Tensor("PROBABILITIES", probabilities.numpy().astype(np.float32))
            pred_tensor = pb_utils.Tensor("PREDICTION", predictions.numpy().astype(np.int32))
            score_tensor = pb_utils.Tensor("TOXICITY_SCORE", toxicity_scores.numpy().reshape(-1, 1).astype(np.float32))
            
            # Создаем ответ
            response = pb_utils.InferenceResponse(
                output_tensors=[logits_tensor, probs_tensor, pred_tensor, score_tensor]
            )
            responses.append(response)
        
        return responses

    def finalize(self):
        print("Model finalized - cleaning up resources")
