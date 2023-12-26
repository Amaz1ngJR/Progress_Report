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
将学习率调小至0.0001出现过拟合
```python
bash_lr = 0.0001 学习率
```
测试单张图的输出如下
```
mean_dice 1.0000000 mean_hd95 0.000000
```
从mean_dice 为1 说明是过拟合的了
可视化的输出

![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/042c7fb7-1da6-49ae-84ef-6d3fc6051d1d)

# 23/12/21
输入的数据生成代码
```python
def npz():
    #原图像路径
    path = r'/home/yjr/Swin-Unet/data/Synapse/images/*.png'
    #项目中存放训练所用的npz文件路径
    path2 = r'/home/yjr/Swin-Unet/data/Synapse/train_npz/'
    
    for i, img_path in enumerate(glob.glob(path)):
        # 读入图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #转成RGB
        # 读入标签
        label_path = img_path.replace('images', 'labels')
        label = cv2.imread(label_path, flags = cv2.IMREAD_GRAYSCALE)
        print("Image shape:", image.shape) (512,512,3)
        print("Label shape:", label.shape) (512,512)
        label[label != 255] = 1  # 将非白色像素设置为1
        label[label == 255] = 0  # 将白色像素设置为0

        img_name = os.path.splitext(os.path.basename(img_path))[0]        
        # count_label_255 = np.sum(label == 255)#白色
        # count_label_not_255 = np.sum(label != 255)#黑色
        # 保存npz
        np.savez(os.path.join(path2, f'{img_name}.npz'), image=image, label=label)
        print('------------', i)
        # print(f"Number of pixels with label 255: {count_label_255}")
        # print(f"Number of pixels with label not 255: {count_label_not_255}")
npz()
```

dataloader
```python
# 初始化数据集 对图片进行随机旋转、翻转 读入image(512,512,3)->(3,512,512) 将label转为long
db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
#.................
trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
```
loss：

```python
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(num_classes)
```

model

debug了下图模型的相关代码
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/12f48350-1906-4211-95e1-db9b73f93de6)
```python
swin_transformer_unet_skip_expand_decoder_sys.py
def forward(self, x):
        x, x_downsample = self.forward_features(x) #x:[1,3,512,512]=>=>=>[1,256,768]  #x:[1,256,768] x_downsample<x>(4,x)
        x = self.forward_up_features(x,x_downsample) #x:[1,256,768] =>[1,16384,96]
        x = self.up_x4(x) #x:[1,16384,96] =>[1,2,512,512]

        return x
```
# 23/12/22

CrossEntropyLoss()

![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/7b5fe368-ac85-4ced-89f7-8f096f051af3)
BCEWithLogitsLoss()/nn.BCELoss

![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/549cd873-4898-48fb-9113-caf4a9f9b7c3)

```
base_lr=0.00005  max_epochs=100  loss->0.48 epoch_99.th mean_dice=1
```
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/07f322d2-d288-4d0e-8017-696eed269123)
```
base_lr=0.00005  max_epochs=300 epoch_199.th loss->0.48 mean_dice=1
```
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/6defa587-dd35-4ebd-a481-e55e49ca0546)
```
base_lr=0.00005  max_epochs=300 epoch_249.th loss->0.48 mean_dice=1
```
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/f2b2b0f0-e028-4333-94de-73c587359c47)

```
base_lr=0.0001  max_epochs=100  epoch_99.th loss->0.40  mean_dice=1
```
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/79d88836-f47a-4954-abb0-ddbaeecfa7ec)

```
base_lr=0.0001  max_epochs=200  epoch_149.th loss->0.36  mean_dice=1
```
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/46da2b5f-bce1-4882-884f-b8e9ceec9c89)

```
base_lr超过0.005 epoch149全黑
```
output和label_batch对不上 修改以后

取消掉对dataloder的旋转
```
base_lr = 0.0005 epoch=2400 loss->17.0收敛
```
![W89%SX64D%Z4Y0WUH{3D(EH](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/acee707d-c37b-4667-acbc-7f589d64d6c0)

# 23/12/26
```python

```
将输入的图片进行分块，使用相同的loss分别计算 当loss降到20左右
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/b154730b-8d4b-4ff2-ae20-9d34f6d952e3)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/517b57d9-bd44-43b0-a788-a525c5d4ab83)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/bc8438e3-c589-4070-b8bf-61bd89062ea4)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/d5eb1eb5-f535-4312-bf05-89d4dae567a6)




