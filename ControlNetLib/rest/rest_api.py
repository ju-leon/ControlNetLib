from fastapi import FastAPI, APIRouter
from fastapi import File, UploadFile, HTTPException
from typing import Optional, Callable
import io
from starlette.responses import StreamingResponse

import numpy as np
import cv2
import torch
from typing import Optional

class RestAPI():
    def __init__(self, model: torch.nn.Module, preprocessor: Optional[Callable] = None, postprocessor: Optional[Callable] = None):
        """
        Initialize a RestAPI for image generation microservice.

        Args:
            model (torch.nn.Module): A model to do inference
            preprocessor (Optional[Callable]): Processing applied to the image before inference. If None, no preprocessing will be applied
            postprocessor (Optional[Callable]): Processing applied to the result after inference. If None, no postprocessing will be applied
        """
        super().__init__()

        self.model = model

        if preprocessor is None:
            preprocessor = lambda x: x

        if postprocessor is None:
            postprocessor = lambda x: x

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.router = APIRouter()

    def register_route(self, url: str, function: Callable, methods: Optional[list] = None):
        """
        Register a route to the router.

        Args:
            url (str): The URL for the route
            function (Callable): The function to be called when the route is accessed
            methods (Optional[list]): The list of HTTP methods that the route accepts. If None, all methods are accepted
        """
        self.router.add_api_route(url, function, methods=methods)
