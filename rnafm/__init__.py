from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import RNAFMConfig, build_config
from .convert import convert, convert_ckpt
from .model import RNAFMForMaskedLM, RNAFMModel
from .tokenizer import RNAFMTokenizer

__all__ = [
    "RNAFMConfig",
    "RNAFMModel",
    "RNAFMForMaskedLM",
    "RNAFMTokenizer",
    "convert",
    "convert_ckpt",
    "build_config",
]


AutoConfig.register("RNAFM", RNAFMConfig)
AutoModel.register(RNAFMConfig, RNAFMModel)
AutoTokenizer.register("RNAFMTokenizer", RNAFMTokenizer)
