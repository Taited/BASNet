import os
from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from PIL import Image
import glob

from data_loader import (RescaleT, SalObjDataset, 
                         ToTensorLab, PadDict, HighContrast)

from model import BASNet


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name, pred, s_dir, d_dir, padding=50):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(
        os.path.join(s_dir, image_name))
    imo = im.resize((image.shape[1] + 2 * padding,
                     image.shape[0] + 2 * padding),
                    resample=Image.Resampling.BILINEAR)
    if padding > 0:
        width, height = image.shape[1], image.shape[0]
        left = padding
        top = padding
        right = width + left
        bottom = height + top
        imo = imo.crop((left, top, right, bottom))
    save_path = os.path.join(d_dir, image_name)
    save_dir = os.path.dirname((save_path))
    if not os.path.exists(save_dir):
        os.makedirs((save_dir))
    imo.save(save_path)


if __name__ == '__main__':
    # --------- 1. get image path and name ---------
    
    image_dir = 'downloads/products'
    prediction_dir = 'downloads/segmentation'
    model_dir = './saved_models/basnet_bsi/basnet.pth'
    
    img_name_list = glob.glob(os.path.join(image_dir, '**/*.png'), recursive=True)
    # --------- 2. dataloader ---------
    #1. dataload
    padding = 50
    test_salobj_dataset = SalObjDataset(
        root_dir=image_dir,
        img_name_list = img_name_list, 
        lbl_name_list = [],
        transform=transforms.Compose([
            PadDict(padding), HighContrast(),
            RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=32,
        shuffle=False, num_workers=2, pin_memory=True)
    
    # --------- 3. model define ---------
    print("...load BASNet...")
    net = BASNet(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    with torch.no_grad():
        # --------- 4. inference for each image ---------
        with tqdm(total=len(test_salobj_dataloader)) as pbar:
            for i_test, data_test in enumerate(test_salobj_dataloader):
                inputs_test = data_test['image']
                inputs_test = inputs_test.type(torch.FloatTensor)
            
                if torch.cuda.is_available():
                    inputs_test = inputs_test.cuda()
            
                d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
            
                # normalization
                for i in range(len(data_test['path'])):
                    pred = d1[i, 0, :, :]
                    pred = normPRED(pred)
                    pred = transforms.CenterCrop(256)(pred)
                    # save results to test_results folder
                    save_output(data_test['path'][i],
                        pred, image_dir, prediction_dir, padding)
                pbar.update(1)
