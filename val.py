import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import lraspp_mobilenetv3_large

from tqdm import tqdm


class pred:
    def __init__(self) -> None:
        weights_path = "./save_weights/model_300.pth"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        palette_path = "./palette_MasTr1325.json"
        print("using {} device.".format(device))
        self.model = lraspp_mobilenetv3_large(num_classes=4)

        # load weights
        weights_dict = torch.load(weights_path, map_location='cpu')['model']
        self.model.load_state_dict(weights_dict)
        self.model.to(device)
        self.model.eval()  # 进入验证模式
        self.device = device

        with open(palette_path, "rb") as f:
            pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
        self.pallette = pallette
        pass
    def work(self,img_path):
        # load image
        original_img = Image.open(img_path)
        srcSize = original_img.size

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(520),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = self.model(img.to(self.device))
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction[np.where(prediction==3)] = 0
            mask = Image.fromarray(prediction)
            mask.putpalette(self.pallette)
            mask = mask.resize((512,384))
            #mask.save("./TestImgs/test_result_100p.png")
        return mask


def main():
    srcImgPATH = r"D:\datasets\mods_2\sequences"
    saveImgPATH = r"D:\datasets\mods_2\segment_pred\mynet"
    
    predh = pred()
    for subdir in os.listdir(srcImgPATH):
        savesubDir = os.path.join(saveImgPATH,subdir)
        os.makedirs(savesubDir,exist_ok=True)
        for imgname in tqdm(os.listdir(os.path.join(srcImgPATH,subdir,"frames"))):
            if "L" in imgname:
                saveImgName = os.path.join(savesubDir,imgname.replace('.jpg','.png'))
                srcImgpath = os.path.join(srcImgPATH,subdir,"frames",imgname)
                mask_p = predh.work(srcImgpath)
                mask_p.save(saveImgName)




if __name__ == '__main__':
    main()
