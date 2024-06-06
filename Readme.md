# 使用教學

常用指令
```
python .\yolov5\train.py --img-size 10 --batch 128 --epochs 100 --data latte.yaml --weights best.pt
```

教學網站
https://officeguide.cc/pytorch-yolo-v5-object-egg-detection-models-tutorial-examples/#google_vignette

標註工具閃退問題
https://blog.csdn.net/m0_74232237/article/details/130985914

tensorboard --logdir yolov5\runs\train

# 如何使用LabelTool

1. run Preprocess.py
2. run Scoring_Tool.py

```
ForTestingImage.csv存放圖片分數
record.csv存放使用者的選擇
```