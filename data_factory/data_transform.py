from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def data_transform(name, size):
    name = name.strip().split('+')
    name = [n.strip() for n in name]
    transform = []
    
    # Loading Method:
    if "resize_random_crop" in name:
        transform.extend([
            transforms.Resize(int(size * 8. / 7.)), # 224 -> 256
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5)
            ])
    elif "resize" in name:
        transform.extend([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5)
            ])
    else:
        # "auto"
        transform.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size)
            ])
    
    if "colorjitter" in name:
        transform.append(
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2))

    transform.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)
    return transform
    
