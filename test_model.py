# Primary source
# https://www.rootstrap.com/blog/how-to-build-a-crowd-counting-model-using-csrnet


import sys

import numpy as np

# import h5py
# import scipy.io as io
import PIL.Image as Image
import torch

# from scipy.ndimage import gaussian_filter
# import scipy
# import torchvision.transforms.functional as F
from matplotlib import cm
from matplotlib import pyplot as plt
from torchvision import transforms

from models.csrnet import CSRNet


def main():

    # ===== input arguments and usage ===================================================================

    # input arguments
    model_path = ""
    if len(sys.argv) < 2:  # not enough arguments
        print("Usage:")
        print("     python test_model.py [test_image_path] [model_path](optional)")
        print("     python -m test_model [test_image_path] [model_path](optional)")
        print(
            "  If no model is provided, will use one in current directory or project home directory"
        )
        return
    elif len(sys.argv) == 2:  # only test image argument
        # TODO: add a code block o find a model or print&print if no model found
        model_path = "best_model.pth"
    elif len(sys.argv) == 3:  # test image and model argumunts provided
        model_path = sys.argv[2]

    # check arguments validity
    if not model_path.lower().endswith(".pth"):
        print(f"Model path {model_path} does not end with .pth")
        return
    test_img_path = sys.argv[1]
    if not test_img_path.lower().endswith(".jpg"):
        print(f"Test image path {test_img_path} does not end with .jpg")
        return

    print(f"model: {model_path}")
    print(f"test image: {test_img_path}")

    # =======================================================================================================

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = CSRNet()

    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=False
    )  # TODO: cpu or gpu
    model.load_state_dict(checkpoint)

    print("og image")
    plt.imshow(plt.imread(test_img_path))
    plt.show()

    trans_img = transform(Image.open(test_img_path).convert("RGB"))
    output = model(trans_img.unsqueeze(0))
    print("predicted count: ", int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(
        output.detach()
        .cpu()
        .reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3])
    )
    plt.imshow(temp, cmap=cm.jet)
    plt.show()


if __name__ == "__main__":
    main()
