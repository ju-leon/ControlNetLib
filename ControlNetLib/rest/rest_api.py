from fastapi import FastAPI, APIRouter
from fastapi import File, UploadFile, HTTPException
from typing import Optional, Callable
import io
from starlette.responses import StreamingResponse

import numpy as np
import cv2

class RestAPI():
    def __init__(self, model, preprocessor, postprocessor):
        super().__init__()

        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.router = APIRouter()
        self.router.add_api_route("/get_image", self.get_image, methods=["POST"])

    async def get_image(self, image: UploadFile, center_x: int, center_y: int, size: int) -> dict:        
        """
        A simple API endpoint that always returns a "Hello" message.
        
        Args:
            image (str): An image, which is ignored.
        
        Returns:
            dict: A dictionary with a "Hello" key and a "Demo" value.
        """
        try:
            contents = image.file.read()
        except Exception:
            raise HTTPException(status_code=500, detail='Something went wrong')
        finally:
            image.file.close()
        
        img = np.fromstring(contents, dtype=np.uint8)    
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        preprocessed_image = self.preprocessor(img, (center_x, center_y), size)

        out = self.model.forward(preprocessed_image, "mri tumor image")[0]
        
        postprocessed_image = self.postprocessor(out)

        res, im_png = cv2.imencode(".png", postprocessed_image)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

