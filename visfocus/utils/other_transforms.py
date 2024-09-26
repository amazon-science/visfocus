import torchvision.transforms as transforms
from torch import nn
from PIL import Image

class ResizeWithAspectRatio(transforms.Resize):
    def __init__(self, size, interpolation=Image.Resampling.BILINEAR):
        super(ResizeWithAspectRatio, self).__init__(size, interpolation)
        self.interpolation = interpolation
        self.target_size = size
        self.aspect_ratio = size[1] / size[0]

    def forward(self, img):

        # Calculate the target width and height based on the max size
        width, height = img.size
        target_height, target_width = self.target_size
        target_aspect_ratio = width / height

        if target_aspect_ratio > self.aspect_ratio:
            # 
            new_width = target_width
            new_height = int(new_width / target_aspect_ratio)
        else:
            # 
            new_height = target_height
            new_width = int(new_height * target_aspect_ratio)

        # # Resize the image
        # resized_img = img.resize((new_width, new_height), self.interpolation)

        # aspect_ratio = img.width / img.height
        # new_width = int(self.size[0])
        # new_height = int(new_width / aspect_ratio)
        return img.resize((new_width, new_height), self.interpolation)


class ResizeAndPad(nn.Module):
    def __init__(self, target_size=(1536, 768), interpolation=Image.Resampling.BILINEAR):
        super(ResizeAndPad, self).__init__()
        self.resize_transform = ResizeWithAspectRatio(target_size, interpolation)
        self.target_size = target_size

    def forward(self, img):
        resized_img = self.resize_transform(img)

        new_width, new_height = resized_img.size
        pad_width = max(0, self.target_size[1] - new_width)
        pad_height = max(0, self.target_size[0] - new_height)

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        resized_padded = transforms.functional.pad(resized_img, (pad_left, pad_top, pad_right, pad_bottom), fill=255)
        return resized_padded