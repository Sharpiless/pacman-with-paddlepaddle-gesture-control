from paddlex.cls import transforms
import paddlex
import cv2

base = './data'

train_transforms = transforms.Compose([
    transforms.ResizeByShort(256),
    transforms.RandomCrop(crop_size=224),
    transforms.RandomRotate(rotate_range=30, prob=0.5),
    transforms.RandomDistort(),
    transforms.Normalize()
])

train_dataset = paddlex.datasets.ImageNet(
    data_dir=base,
    file_list='train_list.txt',
    label_list='labels.txt',
    transforms=train_transforms,
    shuffle=True)

model = paddlex.load_model('weights/final')
im = cv2.imread('test.jpg')

print(
    model.evaluate(eval_dataset=train_dataset)
)
