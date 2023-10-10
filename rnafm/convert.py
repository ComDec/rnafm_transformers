import os
import shutil
import sys
from collections import OrderedDict

import danling as dl
import torch
from chanfig import NestedDict

from .config import build_config


def convert_ckpt(ckpt):
    if isinstance(ckpt, str):
        ckpt = dl.load(ckpt)
    ckpt = NestedDict(ckpt)["encoder.encoder"]
    weights = OrderedDict()
    weights["embeddings.word_embeddings.weight"] = ckpt.pop("embed_tokens.weight")
    weights["embeddings.position_embeddings.weight"] = ckpt.pop("embed_positions.weight")
    weights["embeddings.layer_norm.weight"] = ckpt.pop("emb_layer_norm_before.weight")
    weights["embeddings.layer_norm.bias"] = ckpt.pop("emb_layer_norm_before.bias")
    for key, value in ckpt.layers.items():
        weights[f"encoder.layer.{key}.attention.self.query.weight"] = value.pop("self_attn.q_proj.weight")
        weights[f"encoder.layer.{key}.attention.self.query.bias"] = value.pop("self_attn.q_proj.bias")
        weights[f"encoder.layer.{key}.attention.self.key.weight"] = value.pop("self_attn.k_proj.weight")
        weights[f"encoder.layer.{key}.attention.self.key.bias"] = value.pop("self_attn.k_proj.bias")
        weights[f"encoder.layer.{key}.attention.self.value.weight"] = value.pop("self_attn.v_proj.weight")
        weights[f"encoder.layer.{key}.attention.self.value.bias"] = value.pop("self_attn.v_proj.bias")
        weights[f"encoder.layer.{key}.attention.output.dense.weight"] = value.pop("self_attn.out_proj.weight")
        weights[f"encoder.layer.{key}.attention.output.dense.bias"] = value.pop("self_attn.out_proj.bias")
        weights[f"encoder.layer.{key}.attention.LayerNorm.weight"] = value.pop("self_attn_layer_norm.weight")
        weights[f"encoder.layer.{key}.attention.LayerNorm.bias"] = value.pop("self_attn_layer_norm.bias")
        weights[f"encoder.layer.{key}.intermediate.dense.weight"] = value.pop("fc1.weight")
        weights[f"encoder.layer.{key}.intermediate.dense.bias"] = value.pop("fc1.bias")
        weights[f"encoder.layer.{key}.output.dense.weight"] = value.pop("fc2.weight")
        weights[f"encoder.layer.{key}.output.dense.bias"] = value.pop("fc2.bias")
        weights[f"encoder.layer.{key}.LayerNorm.weight"] = value.pop("final_layer_norm.weight")
        weights[f"encoder.layer.{key}.LayerNorm.bias"] = value.pop("final_layer_norm.bias")
    weights["encoder.emb_layer_norm_after.weight"] = ckpt.pop("emb_layer_norm_after.weight")
    weights["encoder.emb_layer_norm_after.bias"] = ckpt.pop("emb_layer_norm_after.bias")
    weights["lm_head.dense.weight"] = ckpt.pop("lm_head.dense.weight")
    weights["lm_head.dense.bias"] = ckpt.pop("lm_head.dense.bias")
    weights["lm_head.layer_norm.weight"] = ckpt.pop("lm_head.layer_norm.weight")
    weights["lm_head.layer_norm.bias"] = ckpt.pop("lm_head.layer_norm.bias")
    weights["lm_head.decoder.weight"] = ckpt.pop("lm_head.weight")
    weights["lm_head.decoder.bias"] = ckpt.pop("lm_head.bias")
    return weights

def convert(path):
    config = build_config(path)
    if os.path.exists(config._name_or_path):
        shutil.rmtree(config._name_or_path)
    shutil.copytree(os.path.join(os.path.dirname(__file__), "template"), config._name_or_path)
    config.save_pretrained(config._name_or_path)
    ckpt = dl.load(path)
    weights = convert_ckpt(ckpt["model"])
    torch.save(weights, os.path.join(config._name_or_path, "pytorch_model.bin"))


if __name__ == "__main__":
    convert(sys.argv[1])
