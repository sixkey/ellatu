from PIL import Image, ImageDraw
from aggdraw import Draw
from typing import Callable, Generic, List, TypeVar

ImageAction = Callable[[Image.Image], Image.Image]

T = TypeVar('T')


class Drawing(Generic[T]):

    def __init__(self, image: Image.Image, data: T):
        self.g = Draw(image)
        self.image = image
        self.dims = self.image.width, self.image.height
        self.data = data


def edit_image(path: str, action: ImageAction) -> None:
    with Image.open(path) as image:
        new_image = action(image)
        new_image.save(path)


def edit_sequence(actions: List[ImageAction]) -> ImageAction:
    def action(image: Image.Image) -> Image.Image:
        for action in actions:
            image = action(image)
        return image
    return action


def expand_to_aspect(ratio: float) -> ImageAction:
    def action(image: Image.Image) -> Image.Image:
        width, height = image.size
        if width / height < ratio:
            new_height = height
            new_width = int(new_height * ratio)
        else:
            new_width = width
            new_height = int(new_width / ratio)
        new_im = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        new_im.paste(image, ((new_width - width) //
                             2, (new_height - height) // 2))
        return new_im
    return action


def expand(total: int = 0, hor: int = 0, ver: int = 0, left: int = 0,
           top: int = 0, right: int = 0, bottom: int = 0) -> ImageAction:
    def action(image: Image.Image) -> Image.Image:
        width, height = image.size
        new_width = width + total * 2 + hor * 2 + left + right
        new_height = height + total * 2 + ver * 2 + top + bottom
        x, y = total + hor + left, total + ver + top
        new_im = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        new_im.paste(image, (x, y))
        return new_im
    return action
