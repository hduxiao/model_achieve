from torch.utils.data import IterableDataset
from glob import glob
from PIL import Image
import numpy as np
import cv2
import os


class ImageTrainDataset(IterableDataset):
    def __init__(self, filepath, upscale_factor=3):
        super(ImageTrainDataset, self).__init__()
        self.filepath = filepath
        self.upscale_factor = upscale_factor
        self.patch_size = 17
        self.stride = 17 - 3 # as per paper(Sec 3.2)

    def __iter__(self):
        exts = ['*.jpg', '*.png']
        files = [file for ext in exts for file in glob(os.path.join(self.filepath, ext))]
        for file in files:
            hr_image = Image.open(file).convert('RGB')
            # calc width, height
            lr_width = hr_image.width // self.upscale_factor
            lr_height = hr_image.height // self.upscale_factor
            hr_width = lr_width * self.upscale_factor
            hr_height = lr_height * self.upscale_factor
            # resize
            hr_image = hr_image.resize((hr_width, hr_height), Image.BICUBIC)
            lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)
            hr_image = np.array(hr_image).astype(np.float32)
            lr_image = np.array(lr_image).astype(np.float32)
            # RGB to YCrCb
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
            # only using y
            hr_y = hr_image[:,:,-3]
            lr_y = lr_image[:,:,-3]
            # as per paper
            # extract sub-images
            # in the training phase
            rows = lr_y.shape[0] # height
            cols = lr_y.shape[1] # width
            r = self.upscale_factor
            for i in range(0, rows - self.patch_size + 1, self.stride):
                for j in range(0, cols - self.patch_size + 1, self.stride):
                    lr_crop = lr_y[i:i + self.patch_size, j:j + self.patch_size]
                    hr_crop = hr_y[i * r:i * r + self.patch_size * r, j * r:j * r + self.patch_size * r]
                    lr_crop = np.expand_dims(lr_crop / 255.0, 0)
                    hr_crop = np.expand_dims(hr_crop / 255.0, 0)
                    yield lr_crop, hr_crop


class ImageValidDataset(IterableDataset):
    def __init__(self, filepath, upscale_factor=3):
        super(ImageValidDataset, self).__init__()
        self.filepath = filepath
        self.upscale_factor = upscale_factor

    def __iter__(self):
        exts = ['*.jpg', '*.png']
        files = [file for ext in exts for file in glob(os.path.join(self.filepath, ext))]
        for file in files:
            hr_image = Image.open(file).convert('RGB')
            # calc width, height
            lr_width = hr_image.width // self.upscale_factor
            lr_height = hr_image.height // self.upscale_factor
            hr_width = lr_width * self.upscale_factor
            hr_height = lr_height * self.upscale_factor
            # resize
            hr_image = hr_image.resize((hr_width, hr_height), Image.BICUBIC)
            lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)
            hr_image = np.array(hr_image).astype(np.float32)
            lr_image = np.array(lr_image).astype(np.float32)
            # RGB to YCrCb
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
            # only using y
            hr_y = hr_image[:,:,-3]
            lr_y = lr_image[:,:,-3]
            hr_y = np.expand_dims(hr_y / 255.0, 0)
            lr_y = np.expand_dims(lr_y / 255.0, 0)
            yield lr_y, hr_y


## https://discuss.pytorch.org/t/bufferedshuffledataset/106681
## https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
#class BufferedShuffleDataset(IterableDataset):
#    r"""Dataset shuffled from the original dataset.

#    This class is useful to shuffle an existing instance of an IterableDataset.
#    The buffer with `buffer_size` is filled with the items from the dataset first. Then,
#    each item will be yielded from the buffer by reservoir sampling via iterator.

#    `buffer_size` is required to be larger than 0. For `buffer_size == 1`, the
#    dataset is not shuffled. In order to fully shuffle the whole dataset, `buffer_size`
#    is required to be greater than or equal to the size of dataset.

#    When it is used with :class:`~torch.utils.data.DataLoader`, each item in the
#    dataset will be yielded from the :class:`~torch.utils.data.DataLoader` iterator.
#    And, the method to set up a random seed is different based on :attr:`num_workers`.

#    For single-process mode (:attr:`num_workers == 0`), the random seed is required to
#    be set before the :class:`~torch.utils.data.DataLoader` in the main process.
#        >>> ds = BufferedShuffleDataset(dataset)
#        >>> random.seed(...)
#        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

#    For multi-process mode (:attr:`num_workers > 0`), the random seed is set by a callable
#    function in each worker.
#        >>> ds = BufferedShuffleDataset(dataset)
#        >>> def init_fn(worker_id):
#        ...     random.seed(...)
#        >>> print(list(torch.utils.data.DataLoader(ds, ..., num_workers=n, worker_init_fn=init_fn)))
#    """

#    def __init__(self, dataset, buffer_size):
#        super(BufferedShuffleDataset, self).__init__()
#        self.dataset = dataset
#        self.buffer_size = buffer_size

#    def __iter__(self):
#        buf = []
#        for x in self.dataset:
#            if len(buf) == self.buffer_size:
#                idx = random.randint(0, self.buffer_size - 1)
#                yield buf[idx]
#                buf[idx] = x
#            else:
#                buf.append(x)
#        random.shuffle(buf)
#        while buf:
#            yield buf.pop()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    filepath = "C:/Users/hduxi/Desktop/SelfExSR/data/BSD100/image_SRF_3"
    imageset = ImageTrainDataset(filepath)

    for lr_image, _ in imageset:
        print(type(lr_image))
        break;

    
    train_loader = DataLoader(imageset, batch_size = 1)

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(type(lr_image))
        print(f"lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        lr = lr_image[0].numpy().transpose(1, 2, 0)
        hr = hr_image[0].numpy().transpose(1, 2, 0)
        print(f"{lr.shape}, {hr.shape}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(lr)
        ax1.set_title("Low Res")
        ax2.imshow(hr)
        ax2.set_title("High Res")
        plt.show()
        break
