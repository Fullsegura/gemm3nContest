# Import required libraries
import torch
from transformers import pipeline  # HuggingFace Transformers for running image-to-text
import uvicorn  # ASGI server to run the FastAPI app
from fastapi import FastAPI, File, UploadFile, Form  # FastAPI tools for handling HTTP requests
from PIL import Image  # Pillow for image handling
from io import BytesIO  # To convert uploaded image bytes into usable format
import time  # Used to measure processing time
import logging  # For suppressing unwanted logs

# Suppress verbose logs from PyTorch inductor
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

# Improve matrix multiplication precision (optional performance tweak)
torch.set_float32_matmul_precision('high')

# Load the Gemma 3n model using Hugging Face pipeline with auto device mapping and bf16 precision
pipe = pipeline(
   "image-text-to-text",  # Type of pipeline task
   model="google/gemma-3n-E2B-it",  # Specific Gemma3n fine-tuned model
   device_map="auto",  # Automatically select CPU/GPU
   torch_dtype=torch.bfloat16,  # Use bfloat16 for faster performance on supported hardware
)

# Attempt to compile the model for faster inference (if supported by backend)
try:
    pipe.model = torch.compile(pipe.model, mode="reduce-overhead", fullgraph=True)
    print("Model compiled successfully.")
except Exception as e:
    print(f"Could not compile model: {e}")

# Initialize FastAPI app
app = FastAPI()

# Define POST endpoint `/describe` that accepts an image + text prompt
@app.post("/describe")
async def describe(prompt: str = Form(...), image: UploadFile = File(...)):
    # Custom instruction appended to user prompt to guide the AI model
    system_prompt = (
        "You are 'Guardian of Mother Earth' Look for signs of deforestation, "
        "toxic pools, unusual sediment in rivers, reporters, indigenous people "
        "in the jungle or large pits."
    )
    full_prompt = prompt + system_prompt

    # Start measuring time
    start_time = time.time()

    # Read and decode uploaded image
    image_bytes = await image.read()
    try:
        image_pil = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Failed to open image: {e}"}

    # Format the input to include both text and image
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]
    }]

    # Run inference with Gemma 3n model
    output = pipe(text=messages, images=[image_pil], max_new_tokens=100)

    # Calculate total time taken
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    # Extract the generated description from model output
    description = output[0]["generated_text"][-1]["content"]

    # Return result as JSON including timing and description
    return {
        "Gemma3n processing time": processing_time,
        "description": f"Gemma3n:E2B processing time: **{processing_time:.2f}s**\n💬 {description}"
    }

# Run the server if this script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Host the API on all interfaces at port 8000