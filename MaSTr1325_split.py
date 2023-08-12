import os
from tqdm import tqdm
import random
import shutil
TRAIN_VAL_RATIO = 0.9
random.seed(0)

srcLableDir = "D:\datasets\MaSTr1325\MaSTr1325_masks_512x384"
saveDir = "D:\datasets\MaSTr1325\MaSTr1325_split_masks_512x384"
setSaveDir = "D:\datasets\MaSTr1325\ImageSets"

if __name__ == "__main__":
    allMaskFiles = os.listdir(srcLableDir)
    #print(allMaskFiles)
    random.shuffle(allMaskFiles)
    #print(allMaskFiles) 
    train_num = int(len(allMaskFiles)*TRAIN_VAL_RATIO)
    train_list = allMaskFiles[:train_num]
    val_list = allMaskFiles[train_num:]

    # train_dst_dir = os.path.join(saveDir,'train')
    # os.makedirs(train_dst_dir)
    # for i in train_list:
    #     srcPath = os.path.join(srcLableDir,i)
    #     dstPath = os.path.join(train_dst_dir,i)
    #     shutil.copy(srcPath,dstPath)

    # val_dst_dir = os.path.join(saveDir,'val')
    # os.makedirs(val_dst_dir)
    # for i in val_list:
    #     srcPath = os.path.join(srcLableDir,i)
    #     dstPath = os.path.join(val_dst_dir,i)
    #     shutil.copy(srcPath,dstPath)


    os.makedirs(setSaveDir,exist_ok=True)
    with open(os.path.join(setSaveDir,'trainval.txt'),'w') as f:
        for i in allMaskFiles:
            text = i.split('.')[0]
            wtext= text+"\n"
            f.write(wtext)

    with open(os.path.join(setSaveDir,'train.txt'),'w') as f:
        for i in train_list:
            text = i.split('.')[0]
            wtext= text+"\n"
            f.write(wtext)    

    with open(os.path.join(setSaveDir,'val.txt'),'w') as f:
        for i in val_list:
            text = i.split('.')[0]
            wtext= text+"\n"
            f.write(wtext)   