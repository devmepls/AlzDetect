from fastai.vision import *
import torch
import torchvision
from fastbook import load_learner
from fastai.vision.all import *
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from pathlib import Path





def test(imgpath, trainpath):
    img = load_image(imgpath)
    path2 = Path(trainpath)
    learn = load_learner(path2,'export.pkl')
    pred, idx, outputs = learn.predict(img)
    print('Predicted class: ', pred)
    return pred


test('C:\\Users\\visha\\Downloads\\AlzheimersAI-master\\AlzheimersAI-master\\data\\train\\test\\idk.jpg', 'C:\\Users\\visha\\Downloads\\AlzheimersAI-master\\AlzheimersAI-master\\data\\train\\test\\export.pkl')
