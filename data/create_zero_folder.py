import sys
import shutil
import os

test_set= sorted(os.listdir('day1'), key = lambda x: int(x.split('.')[0]))
folder_0= sorted(os.listdir('0'), key = lambda x: int(x.split('.')[0]))
folder_1= sorted(os.listdir('1'), key = lambda x: int(x.split('.')[0]))
folder_2= sorted(os.listdir('2'), key = lambda x: int(x.split('.')[0]))
folder_3= sorted(os.listdir('3'), key = lambda x: int(x.split('.')[0]))
folder_4= sorted(os.listdir('4'), key = lambda x: int(x.split('.')[0]))
folder_5= sorted(os.listdir('5'), key = lambda x: int(x.split('.')[0]))


frames_with_detections = folder_5 + folder_4 + folder_3 + folder_2 + folder_1 + folder_0
print(frames_with_detections)

frames_with_zero_vehicles= list(set(test_set)-set(frames_with_detections))


k=0
while k<2:
    for i in frames_with_zero_vehicles:
        src =  '/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/vehicle_number/day1/'+i
        dest = '/home/hi-vision/Ugent/datasets/auto_annotate/New_test_set_Brighton/vehicle_number/0/'
        shutil.copy(src, dest)
        print(i)
        k+=1

print(frames_with_zero_vehicles)
print(len(test_set))
print(len(frames_with_zero_vehicles))
print(len(frames_with_detections))
