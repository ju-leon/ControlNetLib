from fastapi import FastAPI, APIRouter
from typing import Optional, Callable
import numpy as np

class RestAPI():
    def __init__(self, model, preprocessor, postprocessor):
        super().__init__()

        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.router = APIRouter()
        self.router.add_api_route("/hello", self.get_image, methods=["GET"])

    async def get_image(self, image: str) -> dict:        
        """
        A simple API endpoint that always returns a "Hello" message.
        
        Args:
            image (str): An image, which is ignored.
        
        Returns:
            dict: A dictionary with a "Hello" key and a "Demo" value.
        """
        return {"Hello": "Demo"}

