import argparse
from ControlNetLib.inference.control_net import ControlNet
from ControlNetLib.rest.rest_api import RestAPI
from fastapi import FastAPI, APIRouter
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("generation_type", type=str)
    args = parser.parse_args()

    preprocess_func = None
    postprocess_func = None
    service_call = None

    if args.generation_type == "tumor":
        from ControlNetLib.generators.tumor_generation import preprocess, postprocess
        preprocess_func = preprocess.preprocess
        postprocess_func = postprocess.postprocess
        service_call = "generate_tumor"
    elif args.generation_type == "normal":
        from ControlNetLib.generators.alzheimer_generation import preprocess, postprocess
        preprocess_func = preprocess
        postprocess_func = postprocess
    
    model = ControlNet(args.model_path)
    
    rest_api = RestAPI(model, preprocess_func, postprocess_func)
    app.include_router(rest_api.router, prefix=f"/{args.generation_type}")

    print("Starting server ...")

if __name__ == "__main__":
    main()
    uvicorn.run(    
        app, 
        host="0.0.0.0", 
        port=8000
    )
