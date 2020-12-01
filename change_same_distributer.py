import os
from itertools import islice
from pathlib import Path
from shutil import copyfile

import shutil, sys

try:
    os.rmdir('same')
    os.rmdir('change')
except:
    pass

try:
    os.mkdir('same')
    os.mkdir('change')
except:
    pass

# Collect list of files
x = {dir: set(os.listdir(dir)) for dir in ['0','1','2','3', '4', '5']}
# print(x)

prevFile = ''
prev = -1
listOfAllFiles = sorted(os.listdir('test_set'), key = lambda x: int(x.split('.')[0]))
folderNames = ['0','1','2','3', '4', '5']
for (i,file) in enumerate(listOfAllFiles):
    # Check which folder the file is in
    inFolder = 'err'
    for folderName in folderNames:
        if file in x[folderName]:
            inFolder = folderName
            break

    if prev == inFolder:
        status = 'same'
    else:
        status = 'change'

    print(file, prevFile, inFolder, prev, status)
    src = './' + inFolder + '/' + file
    dest = './' + status + '/' + file
    print('Copying ' + src + ' to ' + dest)
    copyfile(src, dest)
    prev = inFolder
    prevFile = file
