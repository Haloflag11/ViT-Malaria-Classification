#!/root/miniconda3/bin/python
import os
import pathlib
import cv2
import numpy as np
from PIL import Image
from enum import Enum

from mindspore.dataset.vision import transforms
from scipy import io
from typing import Dict,Optional


from dataloader.data_loader import DataLoader
from dataloader.model_loader import ModelLoader
import mindspore as ms



class ImageLoader:
    def loading(self):
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        infer_loader = DataLoader()
        trans_infer = [
            transforms.Decode(),
            transforms.Resize([336, 336]),
            transforms.Normalize(mean=mean, std=std),
            transforms.HWC2CHW()
        ]
        dataset_infer = infer_loader.load_infer()
        dataset_infer=dataset_infer.map(operations=trans_infer,
                          input_columns=["image"],
                          num_parallel_workers=1)
        dataset_infer = dataset_infer.batch(1)
        return dataset_infer


class Color(Enum):
    """dedine enum color."""
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def check_file_exist(file_name: str):
    """check_file_exist."""
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File `{file_name}` does not exist.")


def color_val(color):
    """color_val."""
    if isinstance(color, str):
        return Color[color].value
    if isinstance(color, Color):
        return color.value
    if isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    if isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    if isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    raise TypeError(f'Invalid type for color: {type(color)}')


def imread(image, mode=None):
    """imread."""
    if isinstance(image, pathlib.Path):
        image = str(image)

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        check_file_exist(image)
        image = Image.open(image)
        if mode:
            image = np.array(image.convert(mode))
    else:
        raise TypeError("Image must be a `ndarray`, `str` or Path object.")

    return image


def imwrite(image, image_path, auto_mkdir=True):
    """imwrite."""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(image_path))
        if dir_name != '':
            dir_name = os.path.expanduser(dir_name)
            os.makedirs(dir_name, mode=777, exist_ok=True)

    image = Image.fromarray(image)
    image.save(image_path)


def imshow(img, win_name='', wait_time=0):
    """imshow"""
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def show_result(img: str,
                result: Dict[int, float],
                text_color: str = 'green',
                font_scale: float = 8,
                row_width: int = 200,
                show: bool = False,
                win_name: str = '',
                wait_time: int = 0,
                out_file: Optional[str] = None) -> None:
    """Mark the prediction results on the picture."""
    img = imread(img, mode="RGB")
    img = img.copy()
    x, y = 20, row_width
    text_color = color_val(text_color)
    for k, v in result.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    font_scale, text_color,thickness=8)
        y += row_width
    if out_file:
        show = False
        imwrite(img, out_file)

    if show:
        imshow(img, win_name, wait_time)


def index2label():
    data_path='/root/dataset_storage/data/test'
    metafile = os.path.join(data_path, "ILSVRC2012_devkit_t12/data/meta.mat")
    meta = io.loadmat(metafile, squeeze_me=True)['synsets']

    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]

    _, wnids, classes = list(zip(*meta))[:3]
    clssname = [tuple(clss.split(', ')) for clss in classes]
    wnid2class = {wnid: clss for wnid, clss in zip(wnids, clssname)}
    wind2class_name = sorted(wnid2class.items(), key=lambda x: x[0])

    mapping = {}
    for index, (_, class_name) in enumerate(wind2class_name):
        mapping[index] = class_name[0]
    return mapping


# Read data for inference
if __name__ == '__main__':
    dataset_infer=ImageLoader().loading()
    model=ModelLoader().model_loader()
    file_paths = [
        "/root/dataset_storage/data/infer/falciparum/fal(105).jpg",
        "/root/dataset_storage/data/infer/falciparum/fal(100).jpg",
        "/root/dataset_storage/data/infer/uninfected/uni(100).jpg",
        "/root/dataset_storage/data/infer/uninfected/uni(105).jpg",
        "/root/dataset_storage/data/infer/vivax/viv(100).jpg",
        "/root/dataset_storage/data/infer/vivax/viv(105).jpg"
    ]
    for i, (image, path) in enumerate(zip(dataset_infer.create_dict_iterator(output_numpy=True), file_paths)):
        image = image["image"]
        image = ms.Tensor(image)
        # checkpoint
        vit_path = '/root/autodl-tmp/checkpoint/vit_b_16_8-25_241.ckpt'
        prob = model.predict(image)
        label = np.argmax(prob.asnumpy(), axis=1)
        # 直接定义标签映射字典
        mapping = {0: "falciparum", 1: "uninfected", 2: "vivax"}
        output = {int(label): mapping[int(label)]}
        print(output)
        show_result(img=path,
                    result=output,
                    out_file=f'/root/autodl-tmp/res/res{i}.jpg')