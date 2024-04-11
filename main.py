from transformers import MixtralForCausalLM, MixtralConfig
from transformers import MistralForCausalLM, MistralConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import os
import torch
from torch.nn import functional as F
import argparse
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default="/workspace/text-generation-webui2/models/mistralai_Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument('--target_dir', default="./experts")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    model_id = args.model_id
    target_dir = args.target_dir
    configuration = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", trust_remote_code=False)
    mistral_config = MistralConfig(**dict(configuration.to_dict()))
    mistral_config.architectures = ['MistralForCausalLM']
    mistral_models = []
    for expert_ind in range(configuration.num_local_experts):
        mistral_model = MistralForCausalLM(mistral_config)
        mistral_model.lm_head = model.lm_head
        mistral_model.model.embed_tokens = model.model.embed_tokens
        for layer_ind in range(len(mistral_model.model.layers)):
            mistral_model.model.layers[layer_ind].self_attn.q_proj = model.model.layers[layer_ind].self_attn.q_proj
            mistral_model.model.layers[layer_ind].self_attn.k_proj = model.model.layers[layer_ind].self_attn.k_proj
            mistral_model.model.layers[layer_ind].self_attn.v_proj = model.model.layers[layer_ind].self_attn.v_proj
            mistral_model.model.layers[layer_ind].self_attn.o_proj = model.model.layers[layer_ind].self_attn.o_proj
            mistral_model.model.layers[layer_ind].self_attn.rotary_emb = model.model.layers[
                layer_ind].self_attn.rotary_emb
            mistral_model.model.layers[layer_ind].mlp.gate_proj = \
            model.model.layers[layer_ind].block_sparse_moe.experts[expert_ind].w1
            mistral_model.model.layers[layer_ind].mlp.up_proj = model.model.layers[layer_ind].block_sparse_moe.experts[
                expert_ind].w2
            mistral_model.model.layers[layer_ind].mlp.down_proj = \
            model.model.layers[layer_ind].block_sparse_moe.experts[expert_ind].w3
            mistral_model.model.layers[layer_ind].input_layernorm = model.model.layers[layer_ind].input_layernorm
            mistral_model.model.layers[layer_ind].post_attention_layernorm = model.model.layers[
                layer_ind].post_attention_layernorm
            mistral_models.append(mistral_model)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for expert_ind in range(configuration.num_local_experts):
        mistral_models[expert_ind].save_pretrained(os.path.join(target_dir, "mistral_expert_" + str(expert_ind)))


