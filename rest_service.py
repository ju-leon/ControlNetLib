import argparse
from ControlNetLib.inference.control_net import ControlNet
from ControlNetLib.rest.rest_api import RestAPI
from fastapi import FastAPI, APIRouter
import uvicorn
import configparser
import os
from pytorch_lightning import seed_everything

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    model_path = config.get('model','path')
    seed = config.get('model','seed')

    generation_type = config.get('application','type')

    # Seed all random ops to allow reproduciable results
    seed_everything(seed)

    preprocess_func = None
    postprocess_func = None
    service_call = None

    if generation_type == "tumor":
        from ControlNetLib.generators.tumor_generation import preprocess, postprocess
        preprocess_func = preprocess.preprocess
        postprocess_func = postprocess.postprocess
        service_call = "generate_tumor"
    elif generation_type == "alzheimer":
        from ControlNetLib.generators.alzheimer_generation import preprocess, postprocess
        preprocess_func = preprocess
        postprocess_func = postprocess
    
    model = ControlNet(model_path, seed = seed)
    
    rest_api = RestAPI(model, preprocess_func, postprocess_func)
    app.include_router(rest_api.router, prefix=f"/{generation_type}")

if __name__ == "__main__":
    main()
    uvicorn.run(    
        app, 
        host="0.0.0.0", 
        port=8000
    )
