from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from typing import List
import torch

# Custom embedding class for BioClinicalBERT
class BioClinicalBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: str = "cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model_name = model_name

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=16
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = torch.sum(masked_embeddings, 1)
        counted = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counted

        return mean_pooled.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_text(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text([text])[0]
