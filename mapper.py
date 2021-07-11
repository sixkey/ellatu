from typing import Callable, Optional, Tuple, Dict, TypeVar
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from math import pi
import json
from pprint import pprint
import tatsu

Color = str
TileColor = Tuple[Optional[Color], Optional[Color]]


class Map:

    def __init__(self, start: Tuple[int, int] = (0, 0)) -> None:
        self.start = start
        self.tiles: Dict[Tuple[int, int], TileColor] = {}

    def get_bounding_coords(self) -> Tuple[int, int, int, int]:

        min_x, min_y = self.start
        max_x, max_y = self.start

        for (x, y), _ in self.tiles.items():
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)

        return min_x, min_y, max_x, max_y


class Ship:

    def __init__(self, position: Tuple[int, int] = (0, 0)) -> None:
        self.position = position
        self.direction = 0

    def move(self, movement: Tuple[int, int]) -> 'Ship':
        self.position = tcut2(tadd(self.position, movement))
        return self

    def rotate(self, amount: int) -> 'Ship':
        self.direction = (self.direction + amount) % 4
        return self


COLORS: Dict[str, Optional[Color]] = {
    'b': "black",
    'g': "green",
    'l': "blue",
    'r': "red",
    '_': None
}


def get_color(character: str) -> Optional[Color]:
    if character not in COLORS:
        return None
    return COLORS[character]


def fill_from_file(mapp: Map, path: str) -> Map:

    with open(path, "r") as f:
        for row, line in enumerate(f):
            for col, ch in enumerate(line):
                color = get_color(ch.lower())
                border_color: Optional[Color] = None
                if ch == 'S' or ch.isupper():
                    mapp.start = (col, row)
                    border_color = "green"
                mapp.tiles[(col, row)] = (color, border_color)

    return mapp


T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")

###############################################################################
# TUPLE OPERATIONS
###############################################################################


def ewise_tuple(a: Tuple[T, ...], b: Tuple[R, ...],
                f: Callable[[T, R], S]) -> Tuple[S, ...]:
    return tuple(f(ae, be) for ae, be in zip(a, b))

def tcut2(a: Tuple[int, ...]) -> Tuple[int, int]:
    return a[0], a[1]

def tadd(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return ewise_tuple(a, b, lambda x, y: x + y)

def tsub(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return ewise_tuple(a, b, lambda x, y: x - y)


def tmul(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return ewise_tuple(a, b, lambda x, y: x * y)


def tdiv(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[float, ...]:
    return ewise_tuple(a, b, lambda x, y: x / y)


def tint(a: Tuple[float, ...]) -> Tuple[int, ...]:
    return tuple(int(v) for v in a)


def tran_position(pos: Tuple[int, int], dims: Tuple[int, int],
                  tile_width: int, center: Tuple[int, int]) -> Tuple[int, int]:
    vals = tadd(tmul(tsub(pos, center), (tile_width, tile_width)),
                tint(tdiv(dims, (2, 2))))
    return vals[0], vals[1]



###############################################################################
# TRANSFORM
###############################################################################


def rotation_matrix(rot: float) -> np.ndarray:
    c, s = np.cos(rot), np.sin(rot)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


def translation_matrix(pos: Tuple[int, int]) -> np.ndarray:
    return np.array(((1, 0, pos[0]), (0, 1, pos[1]), (0, 0, 1)))


def scale_matrix(scale: float) -> np.ndarray:
    return np.array(((scale, 0, 0), (0, scale, 0), (0, 0, 1)))


def transform(polygon: np.ndarray, pos: Tuple[int, int],
              rot: float, scale: float):

    t, r, s = translation_matrix(
        pos), rotation_matrix(rot), scale_matrix(scale)
    return t.dot(r.dot(s.dot(polygon)))

###############################################################################
## RENDERING
###############################################################################

SHIP = np.array(((0, -1, 1), (1, 1, 1), (-1, 1, 1))).T

def render_map(path: str, mapp: Map, ship: Optional[Ship]):

    width = 800
    height = 500

    left, top, right, bottom = mapp.get_bounding_coords()
    cx, cy = mapp.start

    tile_width = 50

    if right != cx:
        tile_width = min(tile_width, width / 2 / (right - cx))
    if left != cx:
        tile_width = min(tile_width, width / 2 / (cx - left))
    if bottom != cy:
        tile_width = min(tile_width, height / 2 / (bottom - cy))
    if top != cy:
        tile_width = min(tile_width, height / 2 / (cy - top))

    tile_width = int(tile_width)

    dims = (width, height)
    center = (cx, cy)

    with Image.new(mode="RGB", size=dims, color="white") as im:
        g = ImageDraw.Draw(im)

        for pos, (color, border_color) in mapp.tiles.items():
            coord = tsub(tran_position(pos, dims, tile_width, center),
                         (tile_width // 2, tile_width // 2))
            if color or border_color:
                g.rectangle(
                    [coord, tadd(coord, (tile_width, tile_width))],
                    fill=color, width=tile_width//10,
                    outline=border_color)

        if ship is not None:
            rot = ship.direction * pi
            pos = tran_position(ship.position, dims, tile_width, center)
            polygon = transform(SHIP, pos, rot, tile_width / 3).T.tolist()
            res_poly = tuple([tuple(p[:2]) for p in polygon])
            print(res_poly)
            g.polygon(tuple(res_poly), "gray")

        im.save(path)


###############################################################################
## PARSING
###############################################################################

def mapper_parser():
    with open("mapper.ebnf") as f:
        grammar = f.read()
        return tatsu.compile(grammar, asmodel=True)

def get_indentation(line: str) -> int:
    res = 0
    for ch in line:
        if ch != ' ':
            break
        res += 1
    return res

def is_empty(line: str) -> bool:
    return len(line.strip()) == 0


def indent_to_codeblocks(code: str) -> str:
    lines = code.splitlines(False)

    res = ""

    indents = [0]

    for line in lines:
        if is_empty(line):
            continue
        cur_indent = get_indentation(line)

        if cur_indent > indents[-1]:
            indents.append(cur_indent)
            res += '{\n'
        while cur_indent < indents[-1]:
            indents.pop()
            res += '}\n'
        res += line[cur_indent:] + '\n'

    for _ in range(len(indents) - 1):
        res += '}\n'

    return res



def mapper_preprocessor(code: str) -> str:
    return indent_to_codeblocks(code)


if __name__ == "__main__":

    mapper_map = fill_from_file(Map(), "map.txt")
    ship = Ship(mapper_map.start)

    parser = mapper_parser()

    with open("main.mapper") as f:
        code = f.read()
        code = mapper_preprocessor(code)
        print(code)
        model = parser.parse(code)
        pprint(model)


    # render_map("image.png", mapper_map, ship)
