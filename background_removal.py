import cv2
import numpy as np
import os
from pathlib import Path

global k

from os import listdir
import os, random, shutil
from os.path import isfile, join
import time

import pandas as pd

dir=(os.listdir("/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/day1/"))
def file1():
        j =0
        while j <1:
            testnewset1 = dir
            testnewset_sorted1 = sorted(testnewset1)
            res1 = testnewset_sorted1[k]
            res1 =Path(f'/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/day1/{res1}').stem
            res1= '/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/day1/'+str(res1)+'.jpg'
            j+=1
            print(res1)
        return res1

def file2():
        j =0
        while j <1:
            testnewset1 = dir
            testnewset_sorted1 = sorted(testnewset1)
            res1 = testnewset_sorted1[k]
            res2= testnewset_sorted1[k + 1]
            res2= '/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/day1/'+str(res2)
            print(res2)
            j+=1
        return res2

def res2_filename():
        j =0
        while j <1:
            testnewset1 = dir
            testnewset_sorted1 = sorted(testnewset1)
            res1 = testnewset_sorted1[k]
            res2 =Path(f'/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/day1/{res1}').stem
            res2_filename = str(res2)
            print("saving file name as",res2_filename)
            j+=1
        return res2_filename


k_length= dir
print(k_length)
print(len(k_length))

k=0
while k<len(k_length):

    img1= file1()
    img2= file2()
    res2_filename1 =res2_filename()
    res2_filename1 = f'/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/Background_removal_test_set_result/{res2_filename1}.jpg'
    print(res2_filename1)

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    diff = cv2.absdiff(img1, img2)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    th = 1
    imask =  mask>th
    canvas = np.zeros_like(img1, np.uint8)
    canvas[imask] = img1[imask]
    cv2.imwrite(res2_filename1, canvas)

    k+=1
