import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)
from transformers_interpret import SequenceClassificationExplainer
from .config import MODEL_PATH, BASE_MODEL, MAX_LEN
import os
import re

class FakeNewsModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self.explainer = None
        
        self.gen_tokenizer = None
        self.gen_model = None
        
        self.stop_words = set([
            "the", "is", "in", "at", "of", "on", "and", "a", "an", "to", "for", 
            "it", "this", "that", "with", "as", "by", "are", "was", "were", 
            "be", "or", "from", "not", "but", "we", "they", "he", "she", "his", 
            "her", "its", "my", "your", "s", "t", "can", "will", "just", "don", 
            "should", "now", "d", "ll", "m", "re", "ve", "ain", "aren", "couldn"
        ])

        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(MODEL_PATH):
                print(f"Loading Classifier from {MODEL_PATH}...")
                self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
                self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
            else:
                print("Loading base DistilBERT...")
                self.tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)
                self.model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
            
            self.model.to(self.device)
            self.model.eval()


            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def predict(self, text: str):
        if not text:
            return "UNKNOWN", 0.0, [0.0, 0.0]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
        confidence, predicted_class_idx = torch.max(probs, dim=1)
        idx = predicted_class_idx.item()
        labels = ["FAKE", "REAL"] 
        label = labels[idx]
        
        return label, confidence.item(), probs.cpu().numpy()[0]