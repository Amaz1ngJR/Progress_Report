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

# 23/12/25-26
```python
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes  # 2
    batch_size = args.batch_size * args.n_gpu # 2
    # 初始化数据集 对图片进行随机旋转、翻转 读入image(512,512,3)->(3,512,512) 将label转为long
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    #损失函数和优化器初始化
    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    
    # 主训练循环
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # image_batch.size() [1,3,512,512] label_batch.size() [1,512,512]
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # 将输入图像横向和纵向均匀分成四块
            h, w = image_batch.size(2), image_batch.size(3)
            h_quarters, w_quarters = h // 2, w // 2
            image_quarters = [
                image_batch[:, :, :h_quarters, :w_quarters],
                image_batch[:, :, :h_quarters, w_quarters:],
                image_batch[:, :, h_quarters:, :w_quarters],
                image_batch[:, :, h_quarters:, w_quarters:]
            ]

            label_quarters = [
                label_batch[:, :h_quarters, :w_quarters],
                label_batch[:, :h_quarters, w_quarters:],
                label_batch[:, h_quarters:, :w_quarters],
                label_batch[:, h_quarters:, w_quarters:]
            ]

            total_loss = 0.0
            resize_transform = Resize(512)
            for image_quarter, label_quarter in zip(image_quarters, label_quarters):
                image_quarter = resize_transform(image_quarter)
                label_quarter = resize_transform(label_quarter)
                outputs = model(image_quarter)
                outputs = outputs.squeeze(0)
                cv2.imshow("GT", label_quarter.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
                cv2.imshow("outputs", outputs.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
                cv2.waitKey(0)
                # 使用相同的交叉熵损失
                loss_ce = ce_loss(outputs, label_quarter)

                total_loss += loss_ce.item()

                optimizer.zero_grad()
                loss_ce.backward()
                optimizer.step()

            # 取四块损失的平均值作为总体损失
            avg_loss = total_loss / len(image_quarters)

            cv2.imshow("GT", label_batch.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            cv2.imshow("outputs", outputs.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            cv2.waitKey(0)

            loss = 0.4 * torch.tensor(avg_loss, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  #学习率逐渐降低
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

    writer.close()
    return "Training Finished!"
```
将输入的图片进行分块，使用相同的loss分别计算 当loss降到20左右
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/8d353b98-6815-4c8a-a77e-8b09d02c34e2)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/6b91fcbb-d25c-4dbf-be85-5acd5c271ded)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/bc8438e3-c589-4070-b8bf-61bd89062ea4)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/0da81ae4-f142-45a1-b584-0d9d439a4e89)

# 23/12/27
设置RandomCrop使用pytorch自带的
```python
# 随机裁剪输入图像和标签
crop_size = (256, 256)  # 设置裁剪的大小
random_crop = RandomCrop(crop_size)
//------image_batch和label_batch剪切的地方不一致 没法正确计算loss
image_batch = random_crop(image_batch)
label_batch = random_crop(label_batch)
```
尝试固定seed 但是没有用
```python
# 使用相同的种子确保图像和标签在相同的位置被裁剪
seed_ = torch.randint(0, 2**32 - 1, (1,))
torch.manual_seed(seed_.item())
torch.cuda.manual_seed_all(seed_.item())
```
借助gpt写了一个可以设置seed的
```python
def random_crop_images(image1, image2, crop_size=(256, 256), seed=None):
    # 获取图像的大小
    width = 512 
    height = 512

    # 设置种子
    if seed is not None:
        random.seed(seed)

    # 随机生成裁剪框的左上角坐标
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])

    # 裁剪图像张量
    cropped_tensor1 = image1[:, :, top:top + crop_size[1], left:left + crop_size[0]]
    cropped_tensor2 = image2[:, top:top + crop_size[1], left:left + crop_size[0]]

    return cropped_tensor1, cropped_tensor2
```
```python
 # 主训练循环
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # image_batch.size() [1,3,512,512] label_batch.size() [1,512,512]
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # 使用相同的种子确保图像和标签在相同的位置被裁剪
            seed_ = torch.randint(0, 2**32 - 1, (1,))
            resize_transform = Resize(512) # 设置resize大小
            # 使用 crop 进行裁剪
            image_batch, label_batch = random_crop_images(image_batch,
                    label_batch, seed = seed_)
            
            image_batch = resize_transform(image_batch)
            label_batch = resize_transform(label_batch)

            outputs = model(image_batch)
            outputs = outputs.squeeze(0)
            cv2.imshow("GT", label_batch.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            cv2.imshow("outputs", outputs.permute(1,2,0).detach().cpu().numpy().astype(np.uint8) * 255)
            cv2.waitKey(0)
            loss_ce = ce_loss(outputs, label_batch)

            # optimizer.zero_grad()
            # loss_ce.backward()
            # optimizer.step()

            loss = 0.4 * loss_ce 
```
epoch7000多次的结果(loss 30左右 还没收敛)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/95da38d2-2963-4f1e-a428-f4748ba1eebd)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/a5d91dd1-70d8-476b-bf59-b2ced7fbb4b8)
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/e1bb06e4-3944-402b-a990-603458ef45a2)

# 23/12/29
训练5000次
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/e44471cf-f614-4138-9a86-ced2b65a632e)
训练4W次
![image](https://github.com/Amaz1ngJR/Progress_Report/assets/83129567/1488d422-203c-4fa3-ae10-9984e852cfde)
