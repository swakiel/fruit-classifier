import torchvision.transforms as T

# Basic transforms: resize, convert to tensor, normalize
train_transform = T.Compose([
    T.Resize((100, 100)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),  # converts PIL to tensor and scales 0-1
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

val_transform = T.Compose([
    T.Resize((100, 100)),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
