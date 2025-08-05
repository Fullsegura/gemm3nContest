# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# Fullsegura SAS and Twintivo Inc all rights reserved

# Import required libraries
import numpy as np
import cv2  # For image preprocessing (OpenCV)
from threading import Thread  # To run inference asynchronously
from PIL import Image  # For handling and converting image types
import io  # For buffer manipulation (in-memory image processing)
import requests, base64, time  # HTTP requests, encoding images, timing

# Vision-Language Model (VLM) class for processing images + prompts with remote model
class VLM:

    def __init__(self, url, api_key, callback, model_name=None):
        # Determine model name from URL or use explicit parameter
        if model_name is None:
            self.model = url.split("/")[-2:]
            self.model = "/".join(self.model)
        else:
            self.model = model_name

        # If using specific NVIDIA preview endpoints, append chat completion path
        if url in [
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
        ]:
            url = url + "/chat/completions"

        self.url = url  # Final API endpoint
        self.busy = False  # Flag to avoid overlapping requests
        self.reply = ""  # Response placeholder
        self.api_key = api_key  # Optional API key
        self.callback = callback  # Callback function to send results

    # Convert input image (path, PIL, numpy) into JPEG bytes
    def _process_image(self, image):
        """Resize image, encode as jpeg to shrink size and return the raw bytes."""

        if isinstance(image, str):  # Load image from path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):  # If already a PIL image
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray):  # Convert from OpenCV (numpy)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:
            print(f"Unsupported image input: {type(image)}")
            return None

        # Resize image to standard dimensions (optimized for model input)
        image = image.resize((1024, 768))

        # Convert to JPEG in memory
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        return image_bytes

    # Internal method that prepares and sends the image+prompt request to the VLM API
    def _call(self, message, image=None, callback_args={}):
        image_bytes = None
        image_b64 = None
        try:
            # Preprocess image
            image_bytes = self._process_image(image)
            if image_bytes is None:
                self.reply = "Error: Could not process image."
                return

            # Encode to base64 for callback (not used in API request itself)
            image_b64 = base64.b64encode(image_bytes).decode()

            # Prepare data for POST request
            data = {'prompt': message}
            files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}

            # Send request to the VLM inference API
            response = requests.post(self.url, data=data, files=files)
            print(response.status_code)
            print(response.text)
            response.raise_for_status()

            # Extract the description returned by the API
            self.reply = response.json().get("description", "No description found.")
        except requests.exceptions.RequestException as e:
            self.reply = f"AI Model Request failed "
            print(self.reply)
        except Exception as e:
            self.reply = f"AI Model Error"
            print(self.reply)
        finally:
            if callback_args.get("alert"):
                print('Model finished.')
            self.busy = False  # Free model for next request

        # Provide base64 image in the callback (optional usage downstream)
        callback_args["image_b64"] = image_b64 if image_b64 else ""
        self.callback(message, self.reply, **callback_args)  # Trigger user-defined callback with result

    # Public method to invoke inference with message and image (non-blocking via threading)
    def __call__(self, message, image=None, **kwargs):
        if self.busy:
            print("VLM is busy")
            return None
        else:
            self.busy = True
            # Start inference in a new thread to avoid blocking UI or main loop
            Thread(target=self._call, args=(message, image, kwargs)).start()