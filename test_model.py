import sys
import h5py
import scipy.io as io 
import PIL.Image as Image 
import numpy as np 
from matplotlib import pyplot as plt, cm as c 
from scipy.ndimage import gaussian_filter
import scipy 
import torchvision.transforms.functional as F 
from models.csrnet import CSRNet
import torch 
from torchvision import transforms 

def main():
    
#===== input arguments and usage ===================================================================

    # input arguments
    model_path = ""
    if len(sys.argv) < 2: # not enough arguments
        print("Usage:")
        print("     python test_model.py [test_image_path] [model_path](optional)")
        print("     python -m test_model [test_image_path] [model_path](optional)")
        print("  If no model is provided, will use one in current directory or project home directory")
        return
    elif len(sys.argv) == 2: # only test image argument
        # TODO: add a code block o find a model or print&print if no model found 
        model_path = "best_model.pth"
    elif len(sys.argv) == 3: # test image and model argumunts provided
        model_path = sys.argv[2] 
    
    # check arguments validity
    if model_path.lower().endswith(".pth") == False:
        print(f"Model path {model_path} does not end with .pth")
        return
    test_img_path = sys.argv[1]
    if test_img_path.lower().endswith(".jpg") == False:
        print(f"Test image path {test_img_path} does not end with .jpg")
        return 

    print(f"model: {model_path}")
    print(f"test image: {test_img_path}")

# =======================================================================================================

    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    model = CSRNet() 

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False) # TODO: cpu or gpu 
    model.load_state_dict(checkpoint)


if __name__ == "__main__":
    main()
