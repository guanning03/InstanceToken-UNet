import diffusers,torch
from diffusers import StableDiffusionPipeline
from attn.ptp_utils import AttendExciteAttnProcessor, PositionalEmbeddedAttnProcessor, AttentionStore, register_attention_control

torch.manual_seed(0)

model_id = "runwayml/stable-diffusion-v1-5"

model = StableDiffusionPipeline.from_pretrained(model_id).to('cuda:1')
del model.safety_checker

attention_store = AttentionStore(attn_res = 16)
model.attention_store = attention_store
register_attention_control(model.unet, model.attention_store)

prompt = 'a photo of a cat'

image = model(prompt).images[0]

image

aggregate_cross_attention_maps = attention_store.aggregate_attention(from_where=("up", "down", "mid"), is_cross=True)

import pdb; pdb.set_trace()
print(aggregate_cross_attention_maps)