from torchvision import transforms


class DeepCascadeTransform:
    """
    Dataset transformation utility class
    """
    def __init__(self, size):
        self.pipeline = transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size=(size, size), scale=(0.95, 1), antialias=True)
        ])

    def __call__(self, img):
        return self.pipeline(img)
