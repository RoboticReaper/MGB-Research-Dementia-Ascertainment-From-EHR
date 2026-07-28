from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from typing import List
import torch


class BioClinicalBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: str = None, batch_size: int = 32):
        super().__init__()
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.model.eval()

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i: i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
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

            all_embeddings.extend(mean_pooled.cpu().tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_text(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text([text])[0]