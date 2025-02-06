import torch
import numpy as np
from fastapi import FastAPI, APIRouter
from fastapi import File, UploadFile, HTTPException
from typing import Optional, Callable
import io
from starlette.responses import StreamingResponse
import cv2

from typing import Optional, Callable

from .rest_api import RestAPI

class TumorRestAPI(RestAPI):
    def __init__(self, model: torch.nn.Module, preprocessor: Optional[Callable] = None, postprocessor: Optional[Callable] = None):
        super().__init__(model, preprocessor, postprocessor)
    
        self.register_route("/get_image", self.get_image, methods=["POST"])

    async def get_image(self, image: UploadFile, center_x: int, center_y: int, size: int) -> dict:
        """
        API endpoint to generate an image with the tumor segmentation.

        Args:
            image (UploadFile): The input image.
            center_x (int): The x-coordinate of the tumor center.
            center_y (int): The y-coordinate of the tumor center.
            size (int): The size of the output image.

        Returns:
            A png image with the segmentation result, as a StreamingResponse.
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

        with torch.no_grad():
            out = self.model.forward(preprocessed_image, "mri tumor image")[0]
        
        postprocessed_image = self.postprocessor(out)

        _, im_png = cv2.imencode(".png", postprocessed_image)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

