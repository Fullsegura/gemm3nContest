from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import torch
import time
import os

torch.set_float32_matmul_precision('high')
os.environ["HF_HUB_OFFLINE"] = "1"
model_id = "google/gemma-3n-e4b-it"
model = Gemma3nForConditionalGeneration.from_pretrained(model_id, device_map="auto",  torch_dtype=torch.bfloat16,).eval()
processor = AutoProcessor.from_pretrained(model_id)
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}")
else:
    print("CUDA is NOT available. This is the root of the problem.")


#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
url = "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"
image_path1 = url
image_path2 = "ofi2.png" 
image1 = Image.open(image_path1).convert("RGB")
image2 = Image.open(image_path2).convert("RGB")
images = [image1]
prompt = "is there a bike in the images? <image_soft_token> and <image_soft_token>"
model_inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)

input_len = model_inputs["input_ids"].shape[-1]

start_time = time.time()
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=150)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
end_time = time.time()
inference_time = end_time - start_time

print(f"Inference time: {inference_time:.4f} seconds")
print(decoded)
