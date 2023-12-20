# 23/12/19
修改相应代码 跑了测试集 输出
```
mean dice 0.050981 mean hd95 171.000000
```
可视化结果为全黑图片(背景为黑色、流线为绿色(0,255,0))

loss忽大忽小
```
bash_lr = 0.01 学习率
```
# 23/12/20

发现数据集的像素搞反了，用来测试的输入也没更新
重新验证了一下mask是否正确 以及是否生成了正确的数据集
```python
def count():
    #原图像路径
    path = r'/home/yjr/Swin-Unet/data/Synapse/images/*.png'
    count_label_255 = 0
    count_label_not_255 = 0
    for i, img_path in enumerate(glob.glob(path)):
        # 读入标签
        label_path = img_path.replace('images', 'labels')
        label = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)
        # 统计像素个数
        count_label_255 = np.sum(label == 255)
        count_label_not_255 = np.sum(label != 255)
    print(f"Number of pixels with label 255: {count_label_255}")
    print(f"Number of pixels with label not 255: {count_label_not_255}")
count()
```
测试单张图的输出如下
```
mean_dice 0.999922 mean_hd95 0.000000
```
从mean_dice 0.999922接近于1 说明是过拟合的了
但是可视化的输出是一张全绿的图

不知道改哪了 
```
mean_dice 0.0000 mean_hd95 0.000000
```
图片全黑
