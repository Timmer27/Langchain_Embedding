import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional


class SentimentPredictor:
    """SentimentPredictor
    Easily predict the sentiment of a text using the `hun3359/klue-bert-base-sentiment` model.
    
    Args:
        model_name_or_path (str): Path to the pre-trained model or model identifier from Hugging Face.
        device (str or torch.device): The device to run the model on, defaults to 'cpu'.
    """

    def __init__(self, model_name_or_path="hun3359/klue-bert-base-sentiment", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.labels = {
            "0": "분노",
            "1": "툴툴대는",
            "2": "좌절한",
            "3": "짜증내는",
            "4": "방어적인",
            "5": "악의적인",
            "6": "안달하는",
            "7": "구역질 나는",
            "8": "노여워하는",
            "9": "성가신",
            "10": "슬픔",
            "11": "실망한",
            "12": "비통한",
            "13": "후회되는",
            "14": "우울한",
            "15": "마비된",
            "16": "염세적인",
            "17": "눈물이 나는",
            "18": "낙담한",
            "19": "환멸을 느끼는",
            "20": "불안",
            "21": "두려운",
            "22": "스트레스 받는",
            "23": "취약한",
            "24": "혼란스러운",
            "25": "당혹스러운",
            "26": "회의적인",
            "27": "걱정스러운",
            "28": "조심스러운",
            "29": "초조한",
            "30": "상처",
            "31": "질투하는",
            "32": "배신당한",
            "33": "고립된",
            "34": "충격 받은",
            "35": "가난한 불우한",
            "36": "희생된",
            "37": "억울한",
            "38": "괴로워하는",
            "39": "버려진",
            "40": "당황",
            "41": "고립된(당황한)",
            "42": "남의 시선을 의식하는",
            "43": "외로운",
            "44": "열등감",
            "45": "죄책감의",
            "46": "부끄러운",
            "47": "혐오스러운",
            "48": "한심한",
            "49": "혼란스러운(당황한)",
            "50": "기쁨",
            "51": "감사하는",
            "52": "신뢰하는",
            "53": "편안한",
            "54": "만족스러운",
            "55": "흥분",
            "56": "느긋",
            "57": "안도",
            "58": "신이 난",
            "59": "자신하는"
        }

    @torch.no_grad()
    def predict(self, text: str, top_k:int=10):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the top k classes with the highest probabilities
        top_probabilities, top_classes = torch.topk(probabilities, top_k, dim=-1)

        # Prepare the results
        # results = []
        # for i in range(top_k):
        #     class_index = top_classes[0][i].item()
        #     probability = top_probabilities[0][i].item()
        #     label = self.labels[str(class_index)]
        #     results.append({"class": class_index, "label": label, "probability": probability})
        # return results
        labels = [self.labels.get(str(top_classes[0][i].item())) for i in range(top_k)]
        classes = [top_classes[0][i].item() for i in range(top_k)]
        return {'labels': labels, 'classes': classes}

# test = SentimentPredictor()
# result = test.predict(text='오늘 약간 우울해!')
# print('result', result)
# Example usage:
# predictor = SentimentPredictor()
# predictions = predictor.predict("Your input text here")
# for prediction in predictions:
#     print(f"Class: {prediction['class']}, Label: {prediction['label']}, Probability: {prediction['probability']:.4f}")
