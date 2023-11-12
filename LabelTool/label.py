import pandas as pd
import math
import random
import os
import shutil
import numpy as np

# # TODO 可以部份停用某些照片的label

dirpath = r"./LabelTool/backup27"
result = [os.path.join(dirpath, f) for f in os.listdir(
    dirpath) if os.path.isfile(os.path.join(dirpath, f))]

# 11/7 計算標準差 資料標準化 去離群值
data = pd.read_csv('./LabelTool/Score.csv', header=None)
data = data.apply(pd.to_numeric, errors='coerce')
data = pd.DataFrame(data)

# change all of the '.' in data to nan
for i in range(len(data)):
    for j in range(len(data.columns)):
        if (data[j][i] == '.'):
            data[j][i] = math.nan

# 標準化
# # 行平均值、標準差
col_mean = data.mean(axis=0, numeric_only=True)
col_mean = col_mean.values.tolist()

col_std = data.std(axis=0, numeric_only=True)
col_std = col_std.values.tolist()

print(col_mean)
print(col_std)

# 標準化
data = ((data - col_mean) / col_std).round(4)

print(data)

print('---------------------')

# 去離群值， +- 2 個標準差
for i in range(len(data)):
    for j in range(len(data.columns)):
        if ((data[j][i] < -2) or (data[j][i] > 2)):
            data[j][i] = math.nan

# 歸一化
col_max = data.max(axis=0)
col_max = col_max.values.tolist()
col_min = data.min(axis=0)
col_min = col_min.values.tolist()

for i in range(len(data)):
    for j in range(len(data.columns)):
        if (math.isnan(data[j][i])):
            continue
        else:
            data[j][i] = (((data[j][i] - col_min[j]) /
                           (col_max[j] - col_min[j])) * 10).round(4)


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
        if (data[i][j] == '.'):
            continue
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
        if (not (math.isnan(content))):
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
        if (not (math.isnan(content))):
            f.write(str(content))


# 11/12 create label_propability.csv
L_P = pd.read_csv("./LabelTool/Label.csv", header=None, names=['Values'])
label_col = np.array(L_P['Values'].astype(float))
label_col_rounded = np.round(label_col)


target_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
               5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

# count label amount
for i in range(len(label_col_rounded)):
    if (label_col_rounded[i] in target_dict.keys()):
        target_dict[label_col_rounded[i]] += 1

print(target_dict)
# find min label
min_label = 10000
for i in range(len(target_dict)):
    if target_dict[i] != 0:
        if min_label > target_dict[i]:
            min_label = target_dict[i]


# find data length
n = 0
for i in range(len(label_col_rounded)):
    if not (math.isnan(label_col_rounded[i])):
        n += 1

target = []

# use format: min_label / target_dict[i]
for i in range(len(target_dict)):
    try:
        target.append(min_label / target_dict[i])
    except:
        target.append(0)

# save to csv
target = np.array(target)
df = pd.DataFrame(target)
df.to_csv('./LabelTool/label_probability.csv', index=False, header=False)
