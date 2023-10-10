from transformers import EsmForMaskedLM, EsmModel

from .config import RNAFMConfig


class AveragePooler:
    def __call__(self, hidden_states):
        return hidden_states.mean(dim=1)


class ClsPooler:
    def __call__(self, hidden_states):
        return hidden_states[:, 0]


class RNAFMModel(EsmModel):
    config_class = RNAFMConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.pooler
        del self.contact_head
        self.pooler = ClsPooler()
        if self.embeddings.position_embedding_type != "absolute":
            del self.embeddings.position_embeddings


class RNAFMForMaskedLM(EsmForMaskedLM):
    config_class = RNAFMConfig
