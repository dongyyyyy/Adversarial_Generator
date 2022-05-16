from dataloader import *
from models.SelectModel import *
from train import *
from testing import *
from train_makeNoise import *
from train_model_withNoise import *
from noise_analysis import *
import argparse





if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',required=False,help='input model name',default='resnet18')
    parser.add_argument('-pretrained',required=False,help='using pretrained model',default=True)
    parser.add_argument('-mode',required=False,help='train & test',default='None')
    parser.add_argument('-data_path',required=False,help='data path',default='./cifar/')
    parser.add_argument('-random_seed',required=False,help='random_seed',default=2)
    parser.add_argument('-class_num',required=False,help='number of classes',default=10)
    parser.add_argument('-size',required=False,help='image resize',default=224)
    parser.add_argument('-batch_size',required=False,help='mini-batch size',default=512)
    parser.add_argument('-epochs',required=False,help='epochs',default=100)
    parser.add_argument('-stop_epochs',required=False,help='stop epochs',default=5)
    parser.add_argument('-scheduler',required=False,help='learning rate scheduler',default='StepLR')
    parser.add_argument('-loss_function',required=False,help='loss function',default='CE')

    parser.add_argument('-optim', required=False, help='optimizer', default='sgd')
    parser.add_argument('-lr', required=False, help='learning rate', default=0.1)
    parser.add_argument('-augmentation', required=False, help='data augmentation', default=True)

    parser.add_argument('-cuda',required=False,help='cpu(0) or gpu(1)',default=1)
    parser.add_argument('-gpu',required=False,help='gpu num',default=4)

    parser.add_argument('-load_filename',required=False,help='load_filename',default='None')

    args = parser.parse_args()
    if args.mode =='None':
        print(args)
        print('Please add arg -mode train or test!!!')
    elif args.mode =='train':
        print('Train mode')
        train(args)    
    elif args.mode =='test':
        print('Test mode')
    elif args.mode =='makeNoise':
        # train_makeNoise(args)
        train_makeNoise_usingPascalVOC(args)
        # tests(args)
    elif args.mode =='train_noise':
        train_withNoise(args)
    elif args.mode =='Noise_analysis':
        make_noise_analysis(args)
        make_noise_analysis_usingPascalVOC(args)
        
    



