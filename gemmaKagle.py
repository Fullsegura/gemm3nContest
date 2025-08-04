import torch
from transformers import pipeline
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from io import BytesIO
import time
import logging

logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
torch.set_float32_matmul_precision('high')

pipe = pipeline(
   "image-text-to-text",
   model="google/gemma-3n-E2B-it",
   device_map="auto",
   torch_dtype=torch.bfloat16,
)

try:
    pipe.model = torch.compile(pipe.model, mode="reduce-overhead", fullgraph=True)
    print("Model compiled successfully.")
except Exception as e:
    print(f"Could not compile model: {e}")

app = FastAPI()
@app.post("/describe")
async def describe(prompt: str = Form(...), image: UploadFile = File(...)):
    system_prompt = "You are 'Guardian of Mother Earth' Look for signs of deforestation, toxic pools, unusual sediment in rivers, reporters, indigenous people in the jungle or large pits."
    full_prompt = prompt + system_prompt

    start_time = time.time()
    image_bytes = await image.read()
    try:
        image_pil = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Failed to open image: {e}"}
    messages = [
        {
            "role": "user", 
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    output = pipe(text=messages, images=[image_pil], max_new_tokens=100)
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    description = output[0]["generated_text"][-1]["content"]
    return {
        "Gemma3n processing time": processing_time,
        "description": f"Gemma3n:E2B processing time: **{processing_time:.2f}s**\n💬 {description}"
}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
