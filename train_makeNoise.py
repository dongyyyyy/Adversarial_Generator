import torchvision.models as models
from models.SelectModel import *
from torchsummary import summary
import torch.nn.functional as F
import random
import numpy as np
import torch
import csv
import torch.nn as nn
from torch.autograd import Variable
import sys
from tqdm import tnrange, tqdm
import os
import time
import cv2
import multiprocessing
from torch.utils.data import DataLoader

from dataloader import *

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def fgsm_attack(image, epsilon, data_grad):
    # data_grad 의 요소별 부호 값을 얻어옵니다
    sign_data_grad = data_grad.sign()
    # sign_data_grad = data_grad
    # 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
    perturbed_image = image - epsilon*sign_data_grad
    # 값 범위를 [0,1]로 유지하기 위해 자르기(clipping)를 추가합니다
    # perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # 작은 변화가 적용된 이미지를 리턴합니다
    return perturbed_image


def train_makeNoise(args):
    epsilon = 0.1
    threshold = 0.8
    img_save_path = f'/home/eslab/dataset/created_image_withoutNorm_final_noise_withSign/epsilon_{epsilon}/threshold_{threshold}/'
    os.makedirs(img_save_path,exist_ok=True)
    make_noise_number = 10000
    number_of_class = 10
    random.seed(int(args.random_seed))  # seed
    np.random.seed(int(args.random_seed))
    torch.manual_seed(int(args.random_seed))
    load_filename = f'/home/eslab/kdy/Adversarial_Generator/'\
        f'saved_model/use_pretrained_True_models_resnet18/'\
        f'512_sgd_0.1_StepLR/randomSeed_2/imageSize_224_batchSize_512/'\
        f'epochs_100_stop_5/lossFunction_CE/withoutCrop_noneNorm/SupervisedLearning.pth'

    gpu_num = 1
    
    model = select_model(model_name=args.model, pretrained=args.pretrained)
    softmax = torch.nn.Softmax(dim=1)

    # ResNet
    model.fc = nn.Linear(512,args.class_num,bias=False)

    # load model weight
    model.load_state_dict(torch.load(load_filename))

    if args.cuda == 0:
        device = torch.device('cpu')
        model = model.to(device)
    else:
        device = torch.device('cuda')
        model = model.to(device)
        if gpu_num >= 2:
            model = nn.DataParallel(model)
    
    if args.loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transforms
    generate_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            # normalize,
    ])
    for class_num in range(number_of_class):
        current_img_save_path = f'{img_save_path}{class_num}/'
        os.makedirs(current_img_save_path,exist_ok=True)
        for index in range(make_noise_number):
            model.eval()       
            # random image
            # numpy to tensor
            # generated_image = torch.tensor(generated_image) / 255.

            #transform
            # print(generated_image[0,0,0])
            # print(f'class num = {class_num}')

            generated_image = np.random.randint(0,255,size=[224,224,3],dtype=np.uint8)
            generated_image = generate_transform(generated_image)
            generated_image = generated_image.unsqueeze(0)
            generated_image = generated_image.to(device)

            while(True):
                batch_label = torch.tensor([class_num],dtype=torch.long).to(device)
                # print(batch_label)
                # exit(1)
                # print(generated_image.requires_grad)
                generated_image = torch.autograd.Variable(generated_image,requires_grad=True)
                # print(generated_image.max())
                # exit(1)
                # print(generated_image.requires_grad)
                pred = model(generated_image)

                loss = loss_fn(pred, batch_label) #+ beta * norm

                # gradient makes zero
                model.zero_grad()

                # back propagation
                loss.backward()

                data_grad = generated_image.grad.data
                # print(data_grad)
                # exit(1)
                generated_image = fgsm_attack(generated_image,epsilon,data_grad)
                predict = None
                with torch.no_grad():
                    pred = model(generated_image)

                    predict = softmax(pred)

                # 변경된 이미지의 성능 파악
                if predict[0,class_num] > threshold:
                    print(f'current predict probability is {predict}!!')
                    break
                    

            # generated_image = F.interpolate(generated_image,size=(32,32),mode='bilinear',align_corners=False)
            generated_image = generated_image.squeeze(0)
            # generated_image = ((generated_image * torch.tensor(std).unsqueeze(1).unsqueeze(1)) + torch.tensor(mean).unsqueeze(1).unsqueeze(1))
            generated_image = torch.clamp(generated_image, 0, 1)
            generated_image = (generated_image * 255).cpu()
            generated_image = generated_image.type(torch.uint8)
            generated_image = generated_image.numpy()
            generated_image = generated_image.transpose(1,2,0)
            
            cv2.imwrite(f'{current_img_save_path}{index}.png',generated_image)
            print(f'finish {current_img_save_path}{index}.png')

        
def train_makeNoise_usingPascalVOC(args):
    threshold = 0.8
    epsilon = 0.1
    img_save_path = f'/home/eslab/dataset/created_image_withoutNorm_final_withSign/epsilon_{epsilon}/threshold_{threshold}/'
    os.makedirs(img_save_path,exist_ok=True)
    make_noise_number = 10000
    
    number_of_class = 10
    random.seed(int(args.random_seed))  # seed
    np.random.seed(int(args.random_seed))
    torch.manual_seed(int(args.random_seed))
    load_filename = f'/home/eslab/kdy/Adversarial_Generator/'\
        f'saved_model/use_pretrained_True_models_resnet18/'\
        f'512_sgd_0.1_StepLR/randomSeed_2/imageSize_224_batchSize_512/'\
        f'epochs_100_stop_5/lossFunction_CE/withoutCrop_noneNorm/SupervisedLearning.pth'

    gpu_num = 1
    load_dataset = get_pascalvoc_loader(data_dir='/data/ssd2/VOC2012/JPEGImages/',
                           image_size=224,
                           augment=True,
                           random_seed=2,
                           shuffle=True,
                           make_samples=make_noise_number)
    load_dataset = DataLoader(dataset=load_dataset, batch_size=1, pin_memory=True, shuffle=True,
                                    num_workers=1)
    
    model = select_model(model_name=args.model, pretrained=args.pretrained)
    softmax = torch.nn.Softmax(dim=1)

    # ResNet
    model.fc = nn.Linear(512,args.class_num,bias=False)

    # load model weight
    model.load_state_dict(torch.load(load_filename))

    if args.cuda == 0:
        device = torch.device('cpu')
        model = model.to(device)
    else:
        device = torch.device('cuda')
        model = model.to(device)
        if gpu_num >= 2:
            model = nn.DataParallel(model)
    
    if args.loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device)
    

    # define transforms
    generate_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
    ])

    for class_num in range(number_of_class):
        index = 0
        for image in load_dataset:
            current_img_save_path = f'{img_save_path}{class_num}/'
            # os.makedirs(current_img_save_path,exist_ok=True)
            model.eval()       

            # random image
            init_image = image.to(device)
            # save_init_image = init_image.detach().clone()
            # # save_init_image = F.interpolate(save_init_image,size=(32,32),mode='bilinear',align_corners=False)
            # save_init_image = save_init_image.squeeze(0)
            # # init_image = ((generated_image * torch.tensor(std).unsqueeze(1).unsqueeze(1)) + torch.tensor(mean).unsqueeze(1).unsqueeze(1))
            # save_init_image = torch.clamp(save_init_image, 0, 1)
            # save_init_image = (save_init_image * 255).cpu()
            # save_init_image = save_init_image.type(torch.uint8)
            # save_init_image = save_init_image.numpy()
            # save_init_image = save_init_image.transpose(1,2,0)

            # cv2.imwrite(f'{current_img_save_path}{index}_origin.png',save_init_image)
            # print(f'finish {current_img_save_path}{index}_origin.png')
            indexing = 0
            while(True):
                batch_label = torch.tensor([class_num],dtype=torch.long).to(device)
                # print(batch_label)
                # exit(1)
                # print(generated_image.requires_grad)
                init_image = torch.autograd.Variable(init_image,requires_grad=True)
                # print(generated_image.max())
                # exit(1)
                # print(generated_image.requires_grad)
                pred = model(init_image)

                loss = loss_fn(pred, batch_label) #+ beta * norm
                # print('loss = ',loss.item())
                # gradient makes zero
                model.zero_grad()

                # back propagation
                loss.backward()

                data_grad = init_image.grad.data
                # print(data_grad)
                # exit(1)
                init_image = fgsm_attack(init_image,epsilon,data_grad)
                predict = None
                with torch.no_grad():
                    pred = model(init_image)

                    predict = softmax(pred)
                # print('predict : ', predict)
                # 변경된 이미지의 성능 파악
                if predict[0,class_num] > threshold:
                    break
                # else:
                    # print(f'current predict probability is {predict}!!')
                # save_init_image = init_image.detach().clone()
                # # save_init_image = F.interpolate(save_init_image,size=(32,32),mode='bilinear',align_corners=False)
                # save_init_image = save_init_image.squeeze(0)
                # # init_image = ((generated_image * torch.tensor(std).unsqueeze(1).unsqueeze(1)) + torch.tensor(mean).unsqueeze(1).unsqueeze(1))
                # save_init_image = torch.clamp(save_init_image, 0, 1)
                # save_init_image = (save_init_image * 255).cpu()
                # save_init_image = save_init_image.type(torch.uint8)
                # save_init_image = save_init_image.numpy()
                # save_init_image = save_init_image.transpose(1,2,0)
                
                # cv2.imwrite(f'{current_img_save_path}{index}_{indexing}.png',save_init_image)
                # print(f'finish {current_img_save_path}{index}_{indexing}.png')
                indexing += 1
            # init_image = F.interpolate(init_image,size=(32,32),mode='bilinear',align_corners=False)
            save_init_image = init_image.detach().clone()
            # save_init_image = F.interpolate(save_init_image,size=(32,32),mode='bilinear',align_corners=False)
            save_init_image = save_init_image.squeeze(0)
            # init_image = ((generated_image * torch.tensor(std).unsqueeze(1).unsqueeze(1)) + torch.tensor(mean).unsqueeze(1).unsqueeze(1))
            save_init_image = torch.clamp(save_init_image, 0, 1)
            save_init_image = (save_init_image * 255).cpu()
            save_init_image = save_init_image.type(torch.uint8)
            save_init_image = save_init_image.numpy()
            save_init_image = save_init_image.transpose(1,2,0)
            os.makedirs(current_img_save_path,exist_ok=True)
            cv2.imwrite(f'{current_img_save_path}{index}.png',save_init_image)
            print(f'finish {current_img_save_path}{index}.png')
            index += 1
