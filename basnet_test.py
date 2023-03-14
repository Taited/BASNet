import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

from tqdm import tqdm
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name, pred, s_dir, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(
        os.path.join(s_dir, image_name))
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    save_path = os.path.join(d_dir, image_name)
    save_dir = os.path.dirname((save_path))
    if not os.path.exists(save_dir):
        os.makedirs((save_dir))
    imo.save(save_path)


if __name__ == '__main__':
    # --------- 1. get image path and name ---------
    
    image_dir = '/mnt/tedsun/cafidata/dataset/CafiGarments/products/'
    prediction_dir = '/mnt/tedsun/cafidata/dataset/CafiGarments/basnet_results/'
    model_dir = './saved_models/basnet_bsi/basnet.pth'
    
    img_name_list = glob.glob(os.path.join(image_dir, '**/*.jpg'), recursive=True)
    
    # --------- 2. dataloader ---------
    #1. dataload
    test_salobj_dataset = SalObjDataset(
        root_dir=image_dir,
        img_name_list = img_name_list, 
        lbl_name_list = [],
        transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=4,
                                     shuffle=False, num_workers=1)
    
    # --------- 3. model define ---------
    print("...load BASNet...")
    net = BASNet(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # --------- 4. inference for each image ---------
    with tqdm(total=len(test_salobj_dataloader)) as pbar:
        for i_test, data_test in enumerate(test_salobj_dataloader):
        
            # print("inferencing:",img_name_list[i_test].split("/")[-1])
        
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)
        
            if torch.cuda.is_available():
                inputs_test = inputs_test.cuda()
        
            d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
        
            # normalization
            for i in range(len(data_test['path'])):
                pred = d1[i, 0, :, :]
                pred = normPRED(pred)
            
                # save results to test_results folder
                save_output(data_test['path'][i],
                    pred, image_dir, prediction_dir)
            pbar.update(1)
            
