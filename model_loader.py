from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import requests
from io import BytesIO

model_id = "google/medgemma-4b-it"

print("Loading model")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
).to("cpu")

processor = AutoProcessor.from_pretrained(model_id)

print("Loading image")
# image_url = "https://img.medscapestatic.com/pi/meds/ckb/10/16810tn.jpg"
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

print("Loaded image")

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},
            {"type": "image", "image": image}
        ]
    }
]


inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to("cpu", dtype=torch.float32)

input_len = inputs["input_ids"].shape[-1]

print("Inferencing image")
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

print(f"Model loaded and cached successfully.\nOutput: {output}")
