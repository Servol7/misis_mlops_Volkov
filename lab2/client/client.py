import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import argparse


class TritonToxicityClient:
    def __init__(self, url="localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = "toxicity_classifier"
        
    def check_server_ready(self):
        try:
            return self.client.is_server_ready()
        except InferenceServerException as e:
            print(f"Server is not ready: {e}")
            return False
            
    def check_model_ready(self):
        try:
            return self.client.is_model_ready(self.model_name)
        except InferenceServerException as e:
            print(f"Model is not ready: {e}")
            return False
    
    def predict_toxicity(self, texts):
        
        # Создаем входной тензор
        inputs = []
        texts_np = np.array([text.encode('utf-8') for text in texts])
        inputs.append(httpclient.InferInput("TEXT", texts_np.shape, "BYTES"))
        inputs[0].set_data_from_numpy(texts_np)
        
        # Создаем выходные тензоры
        outputs = [
            httpclient.InferRequestedOutput("LOGITS"),
            httpclient.InferRequestedOutput("PROBABILITIES"),
            httpclient.InferRequestedOutput("PREDICTION"),
            httpclient.InferRequestedOutput("TOXICITY_SCORE")
        ]
        
        # Выполняем запрос
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Получаем результаты
            logits = response.as_numpy("LOGITS")
            probabilities = response.as_numpy("PROBABILITIES")
            predictions = response.as_numpy("PREDICTION")
            toxicity_scores = response.as_numpy("TOXICITY_SCORE")
            
            # Формируем результат
            results = []
            for i, text in enumerate(texts):
                result = {
                    "text": text,
                    "logits": logits[i].tolist(),
                    "probabilities": probabilities[i].tolist(),
                    "prediction": int(predictions[i][0]),
                    "toxicity_score": float(toxicity_scores[i][0]),
                    "is_toxic": bool(predictions[i][0] == 1)
                }
                results.append(result)
            
            return results
            
        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            return None
    
    def print_results(self, results):
        """Print formatted results"""
        for i, result in enumerate(results):
            print(f"\n{'='*50}")
            print(f"Text {i+1}: {result['text']}")
            print(f"{'-'*50}")
            print(f"Non-toxic probability: {result['probabilities'][0]:.4f}")
            print(f"Toxic probability: {result['probabilities'][1]:.4f}")
            print(f"Toxicity score: {result['toxicity_score']:.4f}")
            print(f"Prediction: {'TOXIC' if result['is_toxic'] else 'NON-TOXIC'}")
            print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Triton Toxicity Classifier Client')
    parser.add_argument('--url', type=str, default='localhost:8000',
                       help='Triton server URL (default: localhost:8000)')
    parser.add_argument('--text', type=str, nargs='+',
                       help='Texts to classify (can be multiple)')
    
    args = parser.parse_args()
    
    # Пример текстов для классификации
    default_texts = [
        "Это прекрасный день и все хорошо!",
        "Ненавижу всех, вы все уроды!",
        "Спасибо за помощь",
        "Какая же ты тварь."
    ]
    
    texts = args.text if args.text else default_texts
    
    # Создаем клиент
    client = TritonToxicityClient(url=args.url)
    
    # Проверяем сервер
    if not client.check_server_ready():
        print("Server is not ready. Exiting...")
        return
    
    # Проверяем модель
    if not client.check_model_ready():
        print("Model is not ready. Exiting...")
        return
    
    print(f"Server and model are ready. Processing {len(texts)} texts...")
    
    # Выполняем предсказание
    results = client.predict_toxicity(texts)
    
    if results:
        client.print_results(results)
        
        # Сводная статистика
        toxic_count = sum(1 for r in results if r['is_toxic'])
        print(f"\n{'*'*60}")
        print(f"SUMMARY:")
        print(f"Total texts: {len(results)}")
        print(f"Toxic texts: {toxic_count}")
        print(f"Non-toxic texts: {len(results) - toxic_count}")
        print(f"Average toxicity score: {np.mean([r['toxicity_score'] for r in results]):.4f}")
        print(f"{'*'*60}")
    else:
        print("Failed to get predictions")


if __name__ == "__main__":
    main()
