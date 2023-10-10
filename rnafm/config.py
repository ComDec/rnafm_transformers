import os

from transformers import PretrainedConfig


class RNAFMConfig(PretrainedConfig):
    model_type: str = "RNAFM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architectures = ["RNAFMForMaskedLM"]
        self.position_embedding_type = "absolute"


def build_config(path):
    path = os.path.splitext(path)[0]
    name = os.path.basename(path)
    model_type = "RNAFM"
    num_hidden_layers = 12
    hidden_size = 640
    num_attention_heads = 20
    intermediate_size = 5120
    config = RNAFMConfig(
        model_type=model_type,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        pad_token_id=1,
        eos_token_id=2,
        sep_token_id=2,
        mask_token_id=24,
        vocab_size=25,
        emb_layer_norm_before=True,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=1026,
        token_dropout=True,
        initializer_range=0.02,
    )
    config._name_or_path = name
    return config
