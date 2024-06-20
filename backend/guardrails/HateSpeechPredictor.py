import torch
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from typing import Optional


class HateSpeechPredictor:
    """SentimentPredictor
    Easily predict the sentiment of a text using the `hun3359/klue-bert-base-sentiment` model.
    
    Args:
        model_name_or_path (str): Path to the pre-trained model or model identifier from Hugging Face.
        device (str or torch.device): The device to run the model on, defaults to 'cpu'.
    """

    def __init__(self, model_name_or_path='sgunderscore/hatescore-korean-hate-speech', device="cpu"):
        model = BertForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipe = TextClassificationPipeline(
            model = model,
            tokenizer = tokenizer,
            device = -1, # gpu: 0
            return_all_scores = True,
            function_to_apply = 'sigmoid'
        )

    @torch.no_grad()
    def predict(self, text: str, top_k:int=10):
        results = []
        for result in self.pipe(text)[0]:
            if result.get('label') != 'None':
                results.append(result)
        return sorted(results, key=lambda x: x['score'], reverse=True)
        # self.model.eval()
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        # outputs = self.model(**inputs)
        # logits = outputs.logits
        # probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # # Get the top k classes with the highest probabilities
        # top_probabilities, top_classes = torch.topk(probabilities, top_k, dim=-1)

        # Prepare the results
        # results = []
        # for i in range(top_k):
        #     class_index = top_classes[0][i].item()
        #     probability = top_probabilities[0][i].item()
        #     label = self.labels[str(class_index)]
        #     results.append({"class": class_index, "label": label, "probability": probability})
        # return results
        # labels = [self.labels.get(str(top_classes[0][i].item())) for i in range(top_k)]
        # classes = [top_classes[0][i].item() for i in range(top_k)]
        # return {'labels': labels, 'classes': classes}

test = HateSpeechPredictor()
txt = "좋은 하루 보내세요"
# txt = "죽어라"
result = test.predict(text=txt)
print('result', result)
# Example usage:
# predictor = SentimentPredictor()
# predictions = predictor.predict("Your input text here")
# for prediction in predictions:
#     print(f"Class: {prediction['class']}, Label: {prediction['label']}, Probability: {prediction['probability']:.4f}")
