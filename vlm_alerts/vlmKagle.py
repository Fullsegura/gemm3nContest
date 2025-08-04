# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Fullsegura SAS and Twintivo Inc all rights reserved

import numpy as np
import cv2
from threading import Thread
from PIL import Image
import io
import requests, base64, time


class VLM:

    def __init__(self, url, api_key, callback, model_name=None):
        if model_name is None:  # preview VLM APIs have the model in the URL
            self.model = url.split("/")[-2:]
            self.model = "/".join(self.model)
        else:
            self.model = model_name

        if url in [
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
        ]:
            url = url + "/chat/completions"

        self.url = url

        self.busy = False
        self.reply = ""
        self.api_key = api_key
        self.callback = callback

    def _process_image(self, image):
        """Resize image, encode as jpeg to shrink size and return the raw bytes."""

        if isinstance(image, str):  # file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):  # pil image
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray):  # cv2 / np array image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:
            print(f"Unsupported image input: {type(image)}")
            return None

        image = image.resize(
            (1024, 768))
        buf = io.BytesIO()  # temporary buffer to save processed image
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        return image_bytes

    def _call(self, message, image=None, callback_args={}):
        image_bytes = None
        image_b64 = None
        try:
            image_bytes = self._process_image(image)
            if image_bytes is None:
                self.reply = "Error: Could not process image."
                return

            image_b64 = base64.b64encode(image_bytes).decode()
            data = {'prompt': message}
            files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(self.url, data=data, files=files)
            print(response.status_code)
            print(response.text)
            response.raise_for_status()
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
            self.busy = False

        callback_args["image_b64"] = image_b64 if image_b64 else ""
        self.callback(message, self.reply, **callback_args)

    def __call__(self, message, image=None, **kwargs):
        if self.busy:
            print("VLM is busy")
            return None

        else:
            self.busy = True
            Thread(target=self._call, args=(message, image, kwargs)).start()
