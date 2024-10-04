import torch
import diffusers
import torch.nn as nn

from attn.rope import apply_rotary_emb, compute_cis_1d

class InstanceEncoder:
    
    def __init__(self, config):
        self.config = config
        
    def encode(self, prompt, phrase, count, tokenizer, text_encoder):
        text_inputs = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = text_encoder(text_input_ids).last_hidden_state[0]
        # encode phrase
        phrase_inputs = tokenizer(
            phrase,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        phrase_pooled_output = text_encoder(phrase_inputs.input_ids).pooler_output
        instance_embeddings = phrase_pooled_output.repeat(count, 1)
        instance_embeddings = apply_rotary_emb(instance_embeddings, compute_cis_1d(dim=text_embeddings.shape[-1], end_x=count))
        return text_embeddings, instance_embeddings
        