from collections import defaultdict
from ellatu.image_editing import Drawing
from .document_editor import DocumentEditor
from typing import (Any, Callable, DefaultDict, Generator, List, Optional,
                    Tuple, Dict, TypeVar, Union)
from PIL import Image
import aggdraw
import numpy as np
from math import pi
import tatsu
from tatsu.walkers import NodeWalker
import sys
import os
import json
from pprint import pprint

from importlib import resources

Color = str
TileColor = Tuple[Optional[Color], Optional[Color]]

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")

###############################################################################
# Error
###############################################################################


class MapperError(Exception):
    pass


class MapperRuntimeError(MapperError):
    pass


class MapperCompilationError(MapperError):
    pass

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
                  tile_width: int, center: Tuple[float, float]) \
        -> Tuple[float, float]:
    x = ((pos[0] - center[0]) * tile_width + dims[0] // 2)
    y = ((pos[1] - center[1]) * tile_width + dims[1] // 2)
    return x, y

###############################################################################
# MAP AND SHIP
###############################################################################


class Map:

    def __init__(self, start: Tuple[int, int] = (0, 0)) -> None:
        self.start = start
        self.tiles: DefaultDict[Tuple[int, int], TileColor] = \
            defaultdict(lambda: (None, None))

    def get_bounding_coords(self) -> Tuple[int, int, int, int]:

        min_x, min_y = self.start
        max_x, max_y = self.start

        for (x, y), tile_color in self.tiles.items():
            if tile_color == (None, None):
                continue
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)

        return min_x, min_y, max_x, max_y

    def get(self, tile: Tuple[int, int]) -> TileColor:
        return self.tiles[tile]

    def put(self, tile: Tuple[int, int],
            color: Optional[Color] = None,
            border_color: Optional[Color] = None) -> None:

        (new_color, new_border) = self.tiles[tile]
        if color is not None:
            new_color = color
        if border_color is not None:
            new_border = border_color
        self.tiles[tile] = (new_color, new_border)

    def get_tiles(self, start: bool = False) \
            -> Generator[Tuple[Tuple[int, int], TileColor], None, None]:
        yield from self.tiles.items()
        if start:
            yield self.start, (None, 'green')

    def reset(self) -> None:
        self.tiles.clear()

    def shift(self, dx: int, dy: int) -> None:
        temp_tiles = {}
        for (x, y), color in self.tiles.items():
            temp_tiles[(x + dx, y + dy)] = color
        self.tiles.clear()

        for tile, color in temp_tiles.items():
            self.tiles[tile] = color

    def copy(self) -> 'Map':
        result = Map(start=self.start)
        for key, color in self.tiles.items():
            result.tiles[key] = color
        return result


def mapp_diff_old(desired: Map, received: Map) -> Optional[Map]:
    result = received.copy()

    different = False

    ana_from, ana_on = desired, received
    miss_color = 'red'
    diff_color = 'yellow'
    for _ in range(2):
        for tile, (color, _) in ana_from.tiles.items():
            if color is None:
                continue
            (rec_color, _) = ana_on.get(tile)
            if rec_color is None:
                result.put(tile, color=None, border_color=miss_color)
                different = True
                continue
            if rec_color != color:
                result.put(tile, color=None, border_color=diff_color)
                different = True
        ana_from, ana_on = ana_on, ana_from
        miss_color = 'blue'

    return result if different else None

def mapp_diff(desired: Map, received: Map) -> Optional[Map]:
    result = Map()

    for tile, (color, _) in desired.tiles.items():
        if color is None:
            continue
        (rec_color, _) = received.get(tile)
        if rec_color != color:
            result.put(tile, color=color, border_color=None)

    for tile, (color, _) in received.tiles.items():
        if color is None:
            continue
        (rec_color, _) = desired.get(tile)
        if rec_color is None:
            result.put(tile, color='white', border_color=None)

    return result if result.tiles else None


def mov_for_dir(direction: int) -> Tuple[int, int]:
    assert 0 <= direction <= 3, "There are only four directions"
    if direction == 0:
        return (0, -1)
    if direction == 1:
        return (1, 0)
    if direction == 2:
        return (0, 1)
    if direction == 3:
        return (-1, 0)
    return (0, 0)


class Ship:

    def __init__(self, mapp: Map, position: Tuple[int, int] = (0, 0)) -> None:
        self.position = position
        self.direction = 0
        self.mapp = mapp
        self.pendown = True
        self.color = 'black'

    def move_fwd(self, amount: int) -> 'Ship':
        if self.pendown:
            self.put(self.color)

        direction = self.direction
        if amount < 0:
            direction = (direction + 2) % 4

        for _ in range(abs(amount)):
            self.move(mov_for_dir(direction))

            if self.pendown:
                self.put(self.color)

        return self

    def move(self, movement: Tuple[int, int]) -> 'Ship':
        self.position = tcut2(tadd(self.position, movement))
        return self

    def rotate(self, amount: int) -> 'Ship':
        self.direction = (self.direction + amount) % 4
        return self

    def put(self, color: Optional[Color] = None,
            border_color: Optional[Color] = None) -> None:
        self.mapp.put(self.position, color=color, border_color=border_color)


COLORS: Dict[str, Optional[Color]] = {
    'b': "black",
    'g': "green",
    'l': "blue",
    'r': "red",
    '_': None
}

COLORS_BACK: Dict[Color, str] = {}
for char, color in COLORS.items():
    if color:
        COLORS_BACK[color] = char


def get_color(character: str) -> Optional[Color]:
    if character not in COLORS:
        return None
    return COLORS[character]


def fill_from_text_list(mapp: Map, lines: List[str]) -> Map:
    for line in lines:
        col, row, color = line.split()
        mapp.tiles[(int(col), int(row))] = (COLORS[color], None)
    return mapp


def fill_from_text_map(mapp: Map, lines: List[str]) -> Map:
    cx, cy = 0, 0
    for row, line in enumerate(lines):
        for col, ch in enumerate(line):
            color = get_color(ch.lower())
            if ch == 'S' or ch.isupper():
                cx, cy = (col, row)
            mapp.tiles[(col, row)] = (color, None)
    mapp.shift(-cx, -cy)
    return mapp


def fill_from_text(mapp: Map, text: str, sep: Optional[str] = None) -> Map:
    lines = text.splitlines() if sep is None else text.split(sep)
    header = lines[0].strip()
    if header == 'LIST':
        return fill_from_text_list(mapp, lines[1:])  # TODO: yeah slice bad
    return fill_from_text_map(mapp, lines)


def fill_from_file(mapp: Map, path: str, sep: Optional[str] = None) -> Map:
    with open(path, "r") as f:
        return(fill_from_text(mapp, f.read(), sep))


def export_map(mapp: Map, sep: str = '\n') -> str:
    res = 'LIST'
    for (col, row), (color, _) in mapp.tiles.items():
        if color is not None:
            res += sep + f'{col} {row} {COLORS_BACK[color]}'
    return res


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
              rot: float, scale: float) -> np.ndarray:

    t, r, s = translation_matrix(
        pos), rotation_matrix(rot), scale_matrix(scale)
    return t.dot(r.dot(s.dot(polygon)))

###############################################################################
# RENDERING
###############################################################################


SHIP = np.array(((0, -1, 1), (1, 1, 1), (-1, 1, 1))).T


class DrawingData:
    def __init__(self, box: Tuple[int, int, int, int],
                 center: Tuple[float, float], dims: Tuple[int, int],
                 tile_size: int) -> None:
        self.box = box
        self.center = center
        self.tile_size = tile_size
        self.dims = dims


def render_map(path: str, mapp: Map, ship: Optional[Ship] = None) -> None:
    im = draw_map(mapp, ship)
    im.save(path)


def render_multilevel(path: str, dirs: List[Tuple[str, Map]],
                      ship: Optional[Ship] = None) -> None:
    drawing = create_drawing([m for _, m in dirs])
    draw_grid(drawing, color='#eeeeee', bold_color='#bbbbbb')
    for index, (directive, mapp) in enumerate(dirs):
        draw_tiles(drawing, mapp, method=directive, start=index == 0)
    if ship is not None:
        draw_ship(drawing, ship)
    drawing.g.flush()
    drawing.image.save(path)


def draw_grid(drawing: Drawing[DrawingData],
              color: str = 'lightgray', width: int = 2,
              bold_color: str = 'gray') -> None:

    dims = drawing.image.width, drawing.image.height
    tile_size = drawing.data.tile_size
    center = drawing.data.center
    box = drawing.data.box

    coord = tran_position((0, 0), dims, tile_size, center)

    width, height = dims
    left, top, right, bottom = box
    ln_off_x = width / 2 - (width / 2 // tile_size * tile_size)
    if (right-left) % 2 == 0:
        ln_off_x -= tile_size / 2
    ln_off_y = height / 2 - (height / 2 // tile_size * tile_size)
    if (top - bottom) % 2 == 0:
        ln_off_y += tile_size / 2

    ln_off_x = coord[0] - tile_size / 2
    ln_off_y = coord[1] + tile_size / 2

    off_col = coord[0] // tile_size
    off_row = coord[1] // tile_size

    frequencies = [(1, color), (5, bold_color)]
    for freq, line_col in frequencies:
        pen = aggdraw.Pen(line_col, 2)
        for col in range(0, width // tile_size + 1):
            if (col - off_col) % freq == 0:
                x = (col - off_col) * tile_size + ln_off_x
                drawing.g.line((x, 0, x, height), pen)
        for row in range(0, height // tile_size + 1):
            if (row - off_row) % freq == 0:
                y = (row - off_row) * tile_size + ln_off_y
                drawing.g.line((0, y, width, y), pen)


def draw_tiles(drawing: Drawing[DrawingData], mapp: Map,
               method: str = 'rect', start: bool = True) -> None:

    dims = drawing.image.width, drawing.image.height
    tile_width = drawing.data.tile_size
    center = drawing.data.center

    drawing_function = drawing.g.rectangle
    grid_size = tile_width
    border_size = tile_width // 10
    if method == 'point':
        tile_width = tile_width * 3 // 5
        drawing_function = drawing.g.ellipse
    for pos, (color, border_color) in mapp.get_tiles(start):
        if color is None and border_color is None:
            continue
        stroke_color = border_color if border_color is not None else color
        pen = aggdraw.Pen(stroke_color, border_size, 255)
        brush = aggdraw.Brush(color, 255 if color is not None else 0)
        coord = tran_position(
            pos, dims, grid_size, center)
        bound_box = (coord[0] - tile_width // 2, coord[1] - tile_width // 2,
                     coord[0] + tile_width // 2, coord[1] + tile_width // 2)
        stroke_box = (bound_box[0] + border_size // 2,
                      bound_box[1] + border_size // 2,
                      bound_box[2] - border_size // 2,
                      bound_box[3] - border_size // 2)
        drawing_function(bound_box, aggdraw.Pen('', 0, 0), brush)
        drawing_function(stroke_box, pen, aggdraw.Brush('', 0))


def draw_ship(drawing: Drawing[DrawingData], ship: Ship) -> None:
    dims = drawing.image.width, drawing.image.height
    tile_width = drawing.data.tile_size
    center = drawing.data.center

    rot = ship.direction * pi / 2
    ship_pos = tran_position(ship.position, dims, tile_width, center)
    polygon = transform(SHIP, (int(ship_pos[0]), int(ship_pos[1])), rot,
                        tile_width / 3).T.tolist()
    res_poly = []
    for p in polygon:
        res_poly += p[:2]
    pen = aggdraw.Pen("", 0)
    brush = aggdraw.Brush("gray")
    drawing.g.polygon(res_poly, pen, brush)


def master_bounding_box(boxes: List[Tuple[int, int, int, int]]) \
        -> Tuple[int, int, int, int]:
    return (min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes))


def create_drawing(mapps: List[Map]) -> Drawing:
    width = 800
    height = 500

    left, top, right, bottom = master_bounding_box(
        [m.get_bounding_coords() for m in mapps])

    cx, cy = (left + right) / 2, (top + bottom) / 2

    tile_width = 50

    if right != cx:
        tile_width = int(min(tile_width, width / 2 / (right - cx + 1)))
    if left != cx:
        tile_width = int(min(tile_width, width / 2 / (cx - left + 1)))
    if bottom != cy:
        tile_width = int(min(tile_width, height / 2 / (bottom - cy + 1)))
    if top != cy:
        tile_width = int(min(tile_width, height / 2 / (cy - top + 1)))

    tile_width = max(2, int(tile_width)) // 2 * 2

    dims = (width, height)
    center = (cx, cy)

    im = Image.new(mode="RGB", size=dims, color="white")
    return Drawing(im, DrawingData(
        (left, top, right, bottom), center, dims, tile_width
    ))


def draw_map(mapp: Map, ship: Optional[Ship] = None) -> Image.Image:
    drawing = create_drawing([mapp])
    draw_grid(drawing, color='#eeeeee', bold_color='#bbbbbb')
    draw_tiles(drawing, mapp)
    if ship is not None:
        draw_ship(drawing, ship)
    drawing.g.flush()
    return drawing.image


###############################################################################
# PARSING
###############################################################################

MapperParser = Any


def mapper_parser(asmodel: bool = True) -> MapperParser:
    with resources.open_text('ellatu', 'mapper.ebnf') as f:
        grammar = f.read()
        return tatsu.compile(grammar, asmodel=asmodel)


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
            res += ' ' * indents[-1] + '{\n'
            indents.append(cur_indent)
        while cur_indent < indents[-1]:
            indents.pop()
            res += ' ' * indents[-1] + '}\n'

        res += line + '\n'

    for _ in range(len(indents) - 1):
        res += '}\n'

    return res


def mapper_preprocessor(code: str) -> str:
    return indent_to_codeblocks(code)


class ScopeStack:

    def __init__(self, stack: Optional[List[Dict[str, Any]]] = None):
        self.stack: List[Dict[str, Any]] = stack if stack is not None else []

    def lookup(self, symbol: str) -> Optional[Any]:
        for dic in reversed(self.stack):
            if symbol in dic:
                return dic[symbol]
        return None

    def put(self, symbol: str, value: Any, min_layer: int = 0) -> None:
        assert self.stack

        for index in range(len(self.stack) - 1, min_layer - 1, -1):
            if symbol in self.stack[index]:
                self.stack[index][symbol] = value
                return

        self.stack[-1][symbol] = value

    def put_on_last(self, symbol: str, value: Any) -> None:
        assert self.stack
        self.stack[-1][symbol] = value

    def add_scope(self) -> None:
        self.stack.append({})

    def pop_scope(self) -> None:
        self.stack.pop()

    def copy(self, layers: int = 1) -> 'ScopeStack':
        return ScopeStack(self.stack[:layers])


ARIT_OPS: Dict[str, Callable[[int, int], int]] = {
    '&&': lambda a, b: int(a and b),
    '||': lambda a, b: int(a or b),
    '=>': lambda a, b: int(a and not b),

    '<': lambda a, b: int(a < b),
    '<=': lambda a, b: int(a <= b),
    '>': lambda a, b: int(a > b),
    '>=': lambda a, b: int(a >= b),

    '==': lambda a, b: int(a == b),
    '!=': lambda a, b: int(a != b),

    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a // b,
    '%': lambda a, b: a % b,
}

MapperASTNode = Any

MapperInbuiltFunction = Callable[['MapperWalker', List[Any]], Any]


class MapperInbuiltFunctionWrapper():
    def __init__(self, function: MapperInbuiltFunction):
        self.function = function


class MapperWalker(NodeWalker):

    def __init__(self, glob_scope: ScopeStack,
                 inbuilt: Optional[
                     Dict[str, Union[MapperInbuiltFunction, int]]],
                 ship: Ship,
                 mapp: Map, max_out: Optional[int] = None) \
            -> None:
        self.ctx = glob_scope
        self.scope = glob_scope
        if inbuilt:
            for key, value in inbuilt.items():
                if isinstance(value, int):
                    self.scope.put(key, value)
                else:
                    self.scope.put(key, MapperInbuiltFunctionWrapper(value))
        self.ship = ship
        self.mapp = mapp
        self.max_out = max_out
        self.out: List[str] = []

    def walk_object(self, node: MapperASTNode) -> MapperASTNode:
        return node

    def walk__function(self, node: MapperASTNode) -> MapperASTNode:
        self.scope.put(node.name, node)
        return node

    def walk__amn_function(self, node: MapperASTNode) -> MapperASTNode:
        self.walk(node.fun)
        return None

    def walk__codeblock(self, node: MapperASTNode) -> MapperASTNode:
        self.scope.add_scope()
        value = None
        for line in node.lines:
            value = self.walk(line)
            if value is not None:
                break
        self.scope.pop_scope()
        return value

    def walk__if_statement(self, node: MapperASTNode) -> MapperASTNode:
        # print('if')

        if self.walk(node.cond):
            return self.walk(node.body)

        for elif_stmt in node.elifs:
            if self.walk(elif_stmt.cond):
                return self.walk(elif_stmt.body)

        if node.else_block is not None:
            return self.walk(node.else_block)

        return None

    def walk__while_stmt(self, node: MapperASTNode) -> MapperASTNode:

        while self.walk(node.cond):
            result = self.walk(node.body)
            if result is not None:
                lift, value = result
                if lift == 'break':
                    if value is None or value == 1:
                        return None
                    return lift, value - 1
                if lift == 'continue':
                    continue
                return result
        return None

    def walk__for_stmt(self, node: MapperASTNode) -> MapperASTNode:

        self.scope.add_scope()
        gen = self.walk(node.gen)

        return_value = None

        skips = 0

        for value in gen:
            if skips:
                skips -= 1
                continue
            self.scope.put(self.walk(node.var), value)
            result = self.walk(node.body)
            if result is not None:
                lift, resval = result
                if lift == 'break':
                    if resval is None or resval == 1:
                        break
                    return_value = lift, resval - 1
                    break
                if lift == 'continue':
                    continue

                return_value = result
                return return_value

        self.scope.pop_scope()
        return return_value

    def walk__lift(self, node: MapperASTNode) -> MapperASTNode:
        return (node.lift, self.walk(node.value))

    def walk__amnesia_stmt(self, node: MapperASTNode) -> MapperASTNode:
        self.walk(node.stmt)
        return None

    def walk__call(self, node: MapperASTNode) -> MapperASTNode:

        walked_args = [self.walk(v) for v in node.args]

        function = self.scope.lookup(node.name)

        if isinstance(function, MapperInbuiltFunctionWrapper):
            return function.function(self, walked_args)

        if function is None:
            raise MapperRuntimeError(
                f"Function '{node.name}' is not defined")

        if len(function.params) != (len(node.args)):
            number = len(function.params)
            raise pos_args_invnum(node.name, (number, number), len(node.args))

        scope = self.scope
        newscope = self.scope.copy()
        newscope.add_scope()

        for name, value in zip(function.params, node.args):
            newscope.put_on_last(name, self.walk(value))

        self.scope = newscope
        result = self.walk(function.body)
        self.scope = scope

        if result is None:
            return None

        lift, value = result
        if lift != 'return':
            raise MapperRuntimeError(
                f"A uncaught '{lift}' lift tried to escape function")
        return value

    def walk__neg(self, node: MapperASTNode) -> MapperASTNode:
        if node.op == '!':
            return int(not self.walk(node.val))
        elif node.op == '-':
            return - self.walk(node.val)
        raise MapperRuntimeError(f"Invalid negation operator '{node.op}'")

    def walk__range(self, node: MapperASTNode) -> MapperASTNode:
        start = self.walk(node.start)
        end = self.walk(node.end)
        offset = (1 if end >= start else -1)
        step = self.walk(node.step) if node.step is not None else offset

        if (not isinstance(start, int) or not isinstance(end, int)
                or not isinstance(step, int)):
            raise MapperRuntimeError("Values in range were not of type int")
        return range(start, end + offset, step)

    def walk__assigment(self, node: MapperASTNode) -> MapperASTNode:
        value = self.walk(node.value)
        self.scope.put(node.name, value, min_layer=1)
        return value

    def walk__operation(self, node: MapperASTNode) -> MapperASTNode:
        if node.op not in ARIT_OPS:
            raise MapperRuntimeError(f"Unknown operation '{node.op}")
        left_value = self.walk(node.left)
        right_value = self.walk(node.right)
        if not isinstance(left_value, int) or not isinstance(right_value, int):
            raise MapperRuntimeError("Values in arithmetic operation " +
                                     "were not of type int")
        return ARIT_OPS[node.op](left_value, right_value)

    def walk__variable(self, node: MapperASTNode) -> MapperASTNode:
        value = self.scope.lookup(node.name)
        if value is None:
            raise MapperRuntimeError(f"Variable '{node.name}' is undefined")
        return value

    def walk__integer(self, node: MapperASTNode) -> MapperASTNode:
        return int(node.value)


def in_opt_range(value: int,
                 ran: Optional[Tuple[Optional[int], Optional[int]]]) -> bool:
    if ran is None:
        return True

    left, right = ran

    return (left is None or left <= value) \
        and (right is None or right >= value)


def opt_range_str(rng: Optional[Tuple[Optional[int], Optional[int]]]) \
        -> Tuple[str, bool]:
    if rng is None:
        return 'unlimited number of', True

    bottom, top = rng

    if top is not None and bottom is None:
        return f'at most {top}', top != 1
    if bottom is not None and top is None:
        return f'at least {bottom}', bottom != 1
    if bottom == top:
        return str(bottom), bottom != 1
    return f'from {bottom} to {top}', True


def pos_args_invnum(name: str, desired: Optional[Tuple[Optional[int],
                                                       Optional[int]]],
                    got: int):
    rng_string, rng_pl = opt_range_str(desired)
    return MapperRuntimeError(
        f"Function '{name}' takes {rng_string} positional " +
        f"argument{'s' if rng_pl else ''} " +
        f"but {got} were given.")


def argument_check(name: str, number: Optional[Tuple[Optional[int],
                                                     Optional[int]]] = None) \
        -> Callable[[MapperInbuiltFunction], MapperInbuiltFunction]:
    def dec(func: MapperInbuiltFunction) -> MapperInbuiltFunction:
        def wrapper(ctx: MapperWalker, args: List[Any]) -> Any:
            if not in_opt_range(len(args), number):
                raise pos_args_invnum(name, number, len(args))
            return func(ctx, args)
        return wrapper
    return dec


@argument_check('print', number=(1, None))
def map_print(mapper: MapperWalker, args: List[Any]) -> None:
    if mapper.max_out is None or len(mapper.out) < mapper.max_out:
        mapper.out.append(' '.join(str(a) for a in args))


@argument_check('mov', number=(1, 1))
def mov(mapper: MapperWalker, args: List[Any]) -> None:
    mapper.ship.move_fwd(args[0])


@argument_check('rot', number=(1, 1))
def rot(mapper: MapperWalker, args: List[Any]) -> None:
    mapper.ship.rotate(args[0])


COLS = ['black', 'red', 'green', 'blue']
REV_COLS = {c: i for i, c in enumerate(COLS)}


def validate_color(color: int) -> int:
    if color < 0 or color >= len(COLS):
        raise MapperRuntimeError(f'color has to be from 0 to {len(COLS)-1}')
    return color


@argument_check('put', number=(1, 1))
def put(mapper: MapperWalker, args: List[Any]) -> None:
    mapper.ship.put(color=COLS[validate_color(args[0])])


@argument_check('pen', number=(1, 1))
def pen(mapper: MapperWalker, args: List[Any]) -> None:
    mapper.ship.pendown = bool(args[0])


@argument_check('col', number=(1, 1))
def col(mapper: MapperWalker, args: List[Any]) -> None:
    mapper.ship.color = COLS[validate_color(args[0])]


@argument_check('scn', number=(0, 0))
def scn(mapper: MapperWalker, _: List[Any]) -> int:
    col, border = mapper.mapp.get(mapper.ship.position)
    if col is None:
        return -1
    return REV_COLS[col] if col in REV_COLS else -1


INBUILT = {
    'print': map_print,
    'mov': mov,
    'rot': rot,
    'put': put,
    'pen': pen,
    'col': col,
    'scn': scn,
    'BLACK': 0,
    'RED': 1,
    'GREEN': 2,
    'BLUE': 3,
    'RIGHT': 1,
    'LEFT': -1,
    'TRUE': 1,
    'FALSE': 0
}

Model = Any


def compile_code(code: str, parser: Optional[MapperParser] = None) -> Model:
    if parser is None:
        parser = mapper_parser(True)
    processed = mapper_preprocessor(code)
    try:
        return parser.parse(processed)
    except Exception as e:
        raise MapperCompilationError(
            str(e).replace("'{'", 'indent').replace("'}'", 'dedent'))


def compile_codeblocks(codeblocks: List[str],
                       parser: Optional[MapperParser] = None) -> List[Model]:
    return [compile_code(b, parser) for b in codeblocks]


def contains_main(model: Model) -> bool:
    for dec in model['decls']:
        if dec.name == 'main':
            return True
    return False


def run_models(models: List[Model], mapper_map: Optional[Map] = None,
               ship: Optional[Ship] = None,
               function_name: str = 'main') \
        -> Tuple[Any, Map, Ship, List[str]]:
    if mapper_map is None:
        mapper_map = Map()
    if ship is None:
        ship = Ship(mapper_map, mapper_map.start)

    scopestack = ScopeStack()
    scopestack.add_scope()
    walker = MapperWalker(scopestack, INBUILT, ship, mapper_map, max_out=None)
    for model in models:
        for dec in model['decls']:
            walker.walk(dec)
    scopestack.add_scope()

    called_function = scopestack.lookup(function_name)
    if called_function is None:
        raise MapperRuntimeError(f"{function_name} not found")
    try:
        result = walker.walk(called_function.body)
    except RecursionError:
        raise MapperRuntimeError('maximum recursion depth exceeded')
    except ZeroDivisionError as e:
        raise MapperRuntimeError(str(e))

    return result, mapper_map, ship, walker.out


MapperTest = Tuple[str, str]


def generate_level(folder: str, src: str, name: str,
                   prnt_out: bool = False, prnt_ast: bool = False) -> None:

    src = os.path.join(folder, src)
    org_name = name
    name = os.path.join(folder, name)

    tests = ['blank']

    with open(f"{src}.mpp") as f:
        header = f.readline().strip()
        if len(header) >= 5 and header[:5] == '#incl':
            tests = header[5:].strip().split()
        f.seek(0)
        code = f.read()
    if name != src:
        with open(f"{name}.mpp", 'w') as f:
            f.write(code)

    models = [compile_code(code)]
    if prnt_ast:
        pprint(json.loads(str(models[0]).replace("'", '"')))

    test_exports = []

    for index, test in enumerate(tests):
        mapp = Map()
        ship = Ship(mapp)
        start_ship = Ship(mapp)
        out = []
        if test != 'blank':
            _, mapp, ship, out = run_models(models, function_name=test)
        start_mapp = mapp.copy()
        if prnt_out:
            print(out)
        tiles_test = export_map(mapp, sep=';')
        _, mapp, ship, out = run_models(models, mapper_map=mapp)
        if prnt_out:
            print(out)
        render_multilevel(
            f"{name}-{index}-start.png",
            [('rect', start_mapp), ('point', mapp)],
            start_ship)
        tiles_res = export_map(mapp, sep=';')
        render_map(f"{name}-{index}-res.png", mapp, ship)
        test_exports.append((tiles_test, tiles_res))
    save_json(name, org_name, test_exports)


def save_json(path_name: str, org_name: str, tests: List[MapperTest]) -> None:

    filepath = f"{path_name}.json"
    if os.path.isfile(filepath):
        if False and input('override y/N') != 'y':
            return

    prereqs = []
    desc = f'![{org_name}][{path_name}-0-start.png]'

    md_file = f"{path_name}.md"
    if os.path.isfile(md_file):
        with open(md_file) as f:
            header = f.readline()
            prereqs = header[1:].split()
            desc = f.read() + desc
    else:
        with open(md_file, 'w') as f:
            f.write('!\n')

    json_base = f"{path_name}_base.json"
    DocumentEditor().inject({
        'code': org_name,
        'title': org_name,
        'desc': desc,
        'pipeline': 'mapper',
        'tests': tests,
        'prereqs': prereqs
    }).inject_json_file(
        json_base, optional=True
    ).save(filepath)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) < 3 else sys.argv[2]
    generate_level('mapper', sys.argv[1], name)
