from transformers import MixtralForCausalLM, MixtralConfig
import os

configuration = MixtralConfig(
        vocab_size=320,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=64 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=4,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
)

model = MixtralForCausalLM(configuration)

source_model_id = './mixtral-makeup'

model.save_pretrained(source_model_id)