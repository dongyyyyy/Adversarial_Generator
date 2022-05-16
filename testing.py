import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
def check_histogram():
    path = '/home/eslab/dataset/make_image_toAnalysis_noise_withoutSign_pascal/epsilon_0.1/threshold_0.8/0/1/'

    dataset_list = os.listdir(path)
    print(dataset_list)

    origin_noise = path+'origin.png'
    final_noise = path+'final.png'

    origin_image = np.array(Image.open(origin_noise)).flatten()
    final_image = np.array(Image.open(final_noise)).flatten()
    origin_count = np.bincount(origin_image)

    final_count = np.bincount(final_image)
    origin_count = origin_count
    final_count = final_count
   
    print(origin_count)
    print(final_count)

    plt.plot(origin_count,label='random Noise')
    plt.plot(final_count,label='trained Noise')
    
    plt.savefig('pascal_1.png')


    
# check_histogram()