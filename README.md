# project title

by Esteban J Vasco as part of Thesis for graduate studies at CSUF.

## File Structure

```
root/
├── frames/
│   └── videotitle/
│       ├── images/
│       └── annotations.json
├── datasets/
│   └── fish_dataset.py
├── models/
│   └── csrnet.py
├── utils/
│   └── density.py
├── train_model.py
├── test_model.py
└── requirements.txt
```

Video -> Frames -> Datasets (density maps)

Frames & Datasets (density maps) are used to train a model.

Frames are the input and density maps are the output.
