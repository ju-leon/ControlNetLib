import argparse
from ControlNetLib.inference.control_net import ControlNet
from ControlNetLib.rest.tumor_rest_endpoint import TumorRestAPI
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
    model = ControlNet(model_path, seed = seed)    

    preprocess_func = None
    postprocess_func = None
    rest_api = None
    if generation_type == "tumor":
        from ControlNetLib.generators.tumor_generation import preprocess, postprocess
        preprocess_func = preprocess.preprocess
        postprocess_func = postprocess.postprocess
        rest_api = TumorRestAPI(model, preprocess_func, postprocess_func)
    elif generation_type == "alzheimer":
        # Currently not implemented
        raise NotImplementedError

        # Example on how a future generator can be added to support multiple generation types
        from ControlNetLib.generators.alzheimer_generation import preprocess, postprocess
        preprocess_func = preprocess
        postprocess_func = postprocess
        # other rest api ...
    else:
        raise NotImplementedError
    
    app.include_router(rest_api.router, prefix=f"/{generation_type}")

if __name__ == "__main__":
    main()
    uvicorn.run(    
        app, 
        host="0.0.0.0", 
        port=8000
    )
