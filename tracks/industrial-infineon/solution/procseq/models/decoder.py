"""From-scratch Llama-style causal LM sized by a named preset."""
from transformers import LlamaConfig, LlamaForCausalLM

SIZES = {
    "tiny":  dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=4),
    "small": dict(hidden_size=256, intermediate_size=768, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=8),
    "base":  dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=8,
                  num_attention_heads=8, num_key_value_heads=8),
    "large": dict(hidden_size=768, intermediate_size=2304, num_hidden_layers=12,
                  num_attention_heads=12, num_key_value_heads=12),
}

def build_decoder(size, tokenizer, max_position_embeddings=256):
    p = SIZES[size]
    cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_position_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        **p,
    )
    return LlamaForCausalLM(cfg)
