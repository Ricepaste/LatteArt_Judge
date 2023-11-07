import pandas as pd
import math
import random
import os
import shutil
import numpy as np

dirpath = r"./LabelTool/backup27"
result = [os.path.join(dirpath, f) for f in os.listdir(
    dirpath) if os.path.isfile(os.path.join(dirpath, f))]

# 11/7 計算標準差 資料標準化 去離群值
data = pd.read_csv('./LabelTool/Score.csv', header=None)
data = pd.DataFrame(data)

# 行平均值、標準差
col_mean = data.mean(axis=0)
col_mean = col_mean.values.tolist()
col_std = data.std(axis=0)
col_std = col_std.values.tolist()

# 標準化
data = ((data - col_mean) / col_std).round(4)

print(data)

print('---------------------')

# 去離群值， +- 1 個標準差
for i in range(len(data)):
    for j in range(len(data.columns)):
        if ((data[j][i] < -1) or (data[j][i] > 1)):
            data[j][i] = math.nan
            
            
print(data)

data.to_csv('./LabelTool/Label.csv', index=False, header=False)


# 11/4

Train_Size = 0.7
Test_Size = 0.3

# Read the data
data = pd.read_csv('./LabelTool/Label.csv', header=None)
data = np.array(data.values)

average = []

for i in range(len(data)):
    sum = 0
    count = 0
    for j in range(len(data[i])):
        if (math.isnan(data[i][j])):
            continue
        else:
            sum += data[i][j]
            count += 1
    try:
        average.append(sum/count)
    except:
        pass

# print(average)

cut = int(len(average)*Train_Size)

# train = average[:cut]
# test = average[cut:]

# train_data = result[:cut]
# test_data = result[cut:]

average_index_shuffle = [i for i in range(len(average))]
random.shuffle(average_index_shuffle)
train_index_shuffle = average_index_shuffle[:cut]
test_index_shuffle = average_index_shuffle[cut:]

dir1 = os.listdir('./LabelTool/train/images')
dir2 = os.listdir('./LabelTool/val/images')
dir3 = os.listdir('./LabelTool/train/labels')
dir4 = os.listdir('./LabelTool/val/labels')

try:
    if (len(dir1) != 0 or len(dir2) != 0 or len(dir3) != 0 or len(dir4) != 0):
        raise Exception("The folder is not empty")
except:
    for file in os.listdir('./LabelTool/train/images'):
        os.remove('./LabelTool/train/images/' + file)
    for file in os.listdir('./LabelTool/val/images'):
        os.remove('./LabelTool/val/images/' + file)
    for file in os.listdir('./LabelTool/train/labels'):
        os.remove('./LabelTool/train/labels/' + file)
    for file in os.listdir('./LabelTool/val/labels'):
        os.remove('./LabelTool/val/labels/' + file)

# copy the image to the corresponding folder
for m in train_index_shuffle:
    shutil.copy(result[m], './LabelTool/train/images')
    temp = result[m].split('./LabelTool/backup27')
    fuck = temp[1].split('.jpg')
    file_path = '.\\LabelTool\\train\\labels{}.txt'.format(fuck[0])
    content = average[m]
    with open(file_path, 'w+') as f:
        if (not(math.isnan(content))):
            f.write(str(content))

# index_temp = 02

for n in test_index_shuffle:
    temp = result[n].split('./LabelTool/backup27')
    fuck = temp[1].split('.jpg')
    shutil.copy(result[n], './LabelTool/val/images')
    file_path = '.\\LabelTool\\val\\labels{}.txt'.format(fuck[0])
    content = average[n]
    # index_temp += 1
    with open(file_path, 'w+') as f:
        if (not(math.isnan(content))):
            f.write(str(content))
            
