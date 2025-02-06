from ControlNetLib.inference.control_net import ControlNet
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="models/control_sd15_canny.pth")
    parser.add_argument("--out_path", type=str, default="models/serialised_model.pts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ControlNet(args.input, args.seed)
    
    torch_input = (torch.randn(1, 3, 512, 512, device='cuda'), 'prompt')

    # ::triu operator only supported from ONNX > 13
    onnx = torch.onnx.export(model, torch_input, args.out_path, opset_version=14)

if __name__ == "__main__":
    main()