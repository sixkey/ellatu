from collections import defaultdict
from typing import Any, Callable, DefaultDict, List, Optional, Tuple, Dict, TypeVar
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from math import pi
from pprint import pprint
import tatsu
from tatsu.walkers import NodeWalker
import json

Color = str
TileColor = Tuple[Optional[Color], Optional[Color]]

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

        for (x, y), _ in self.tiles.items():
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)

        return min_x, min_y, max_x, max_y

    def get(self, tile: Tuple[int, int]) -> TileColor:
        return self.tiles[tile]

    def put(self, tile: Tuple[int, int],
            color: Optional[Color] = None, border_color: Optional[Color] = None):

        (new_color, new_border) = self.tiles[tile]
        if color is not None:
            new_color = color
        if border_color is not None:
            new_border = border_color
        self.tiles[tile] = (new_color, new_border)

    def get_tiles(self):
        yield from self.tiles.items()
        yield self.start, (None, 'green')

    def reset(self):
        self.tiles.clear()

    def shift(self, dx, dy):
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


def mapp_diff(desired: Map, received: Map) -> Optional[Map]:
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

    def move_fwd(self, amount) -> 'Ship':
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
            border_color: Optional[Color] = None):
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


def fill_from_text(mapp: Map, text: str) -> Map:
    lines = text.splitlines()
    header = lines[0].strip()
    if header == 'LIST':
        return fill_from_text_list(mapp, lines[1:])  # TODO: yeah slice bad
    return fill_from_text_map(mapp, lines)


def fill_from_file(mapp: Map, path: str) -> Map:

    with open(path, "r") as f:
        return(fill_from_text(mapp, f.read()))

def export_map(mapp: Map) -> str:
    res = 'LIST\n'
    for (col, row), (color, _) in mapp.tiles.items():
        if color is not None:
            res += f'{col} {row} {COLORS_BACK[color]}\n'
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
              rot: float, scale: float):

    t, r, s = translation_matrix(
        pos), rotation_matrix(rot), scale_matrix(scale)
    return t.dot(r.dot(s.dot(polygon)))

###############################################################################
# RENDERING
###############################################################################


SHIP = np.array(((0, -1, 1), (1, 1, 1), (-1, 1, 1))).T


def render_map(path: str, mapp: Map, ship: Optional[Ship]):

    width = 800
    height = 500

    left, top, right, bottom = mapp.get_bounding_coords()
    cx, cy = (left + right) / 2, (top + bottom) / 2

    tile_width = 50

    if right != cx:
        tile_width = min(tile_width, width / 2 / (right - cx + 1))
    if left != cx:
        tile_width = min(tile_width, width / 2 / (cx - left + 1))
    if bottom != cy:
        tile_width = min(tile_width, height / 2 / (bottom - cy + 1))
    if top != cy:
        tile_width = min(tile_width, height / 2 / (cy - top + 1))

    tile_width = int(tile_width)

    dims = (width, height)
    center = (cx, cy)

    with Image.new(mode="RGB", size=dims, color="white") as im:
        g = ImageDraw.Draw(im)

        for pos, (color, border_color) in mapp.get_tiles():
            coord = tsub(tran_position(pos, dims, tile_width, center),
                         (tile_width // 2, tile_width // 2))
            g.rectangle(
                [coord, tadd(coord, (tile_width, tile_width))],
                fill=color, width=tile_width//10,
                outline=border_color)

        if ship is not None:
            rot = ship.direction * pi / 2
            pos = tran_position(ship.position, dims, tile_width, center)
            polygon = transform(SHIP, pos, rot, tile_width / 3).T.tolist()
            res_poly = tuple([tuple(p[:2]) for p in polygon])
            g.polygon(tuple(res_poly), "gray")

        im.save(path)


###############################################################################
# PARSING
###############################################################################

def mapper_parser(asmodel=True):
    with open("mapper.ebnf") as f:
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

    def lookup(self, symbol: str):
        for dic in reversed(self.stack):
            if symbol in dic:
                return dic[symbol]
        return None

    def put(self, symbol: str, value: Any) -> None:
        assert self.stack
        index = 0

        while index < len(self.stack) - 1 and symbol not in self.stack[index]:
            index += 1

        self.stack[index][symbol] = value

    def put_on_last(self, symbol: str, value: Any) -> None:
        assert self.stack
        self.stack[-1][symbol] = value

    def add_scope(self):
        self.stack.append({})

    def pop_scope(self):
        self.stack.pop()

    def copy(self, layers: int = 1):
        return ScopeStack(self.stack[:layers])


ARIT_OPS = {
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


class MapperWalker(NodeWalker):

    def __init__(self, glob_scope: ScopeStack,
                 inbuilt: Optional[Dict[str,
                                        Callable[['MapperWalker', List[Any]], Any]]],
                 ship: Ship,
                 mapp: Map) \
            -> None:
        self.ctx = glob_scope
        self.scope = glob_scope
        self.inbuilt = inbuilt if inbuilt is not None else {}
        self.ship = ship
        self.mapp = mapp

    def walk_object(self, node):
        return node

    def walk__function(self, node):
        self.scope.put(node.name, node)
        return node

    def walk__amn_function(self, node):
        self.walk(node.fun)
        return None

    def walk__codeblock(self, node):
        self.scope.add_scope()
        value = None
        for line in node.lines:
            value = self.walk(line)
            if value is not None:
                break
        self.scope.pop_scope()
        return value

    def walk__if_statement(self, node):
        # print('if')

        if self.walk(node.cond):
            return self.walk(node.body)

        for elif_stmt in node.elifs:
            if self.walk(elif_stmt.cond):
                return self.walk(elif_stmt.body)

        if node.else_block is not None:
            return self.walk(node.else_block)

        return None

    def walk__while_stmt(self, node):

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

    def walk__for_stmt(self, node):

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

    def walk__lift(self, node):
        return (node.lift, self.walk(node.value))

    def walk__amnesia_stmt(self, node):
        self.walk(node.stmt)
        return None

    def walk__call(self, node):

        walked_args = [self.walk(v) for v in node.args]

        if node.name in self.inbuilt:
            return self.inbuilt[node.name](self, walked_args)

        function = self.scope.lookup(node.name)
        if function is None:
            raise ValueError(f"function is not defined")

        if len(function.params) != (len(node.args)):
            raise ValueError(f"Invalid number of positional arguments")

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
            raise SyntaxError(f"A uncaught lift other than return tried " +
                              "to escape function")
        return value

    def walk__neg(self, node):
        return int(not self.walk(node.val))

    def walk__range(self, node):
        start = self.walk(node.start)
        end = self.walk(node.end)
        offset = (1 if end >= start else -1)

        step = self.walk(node.step) if node.step is not None else offset
        return range(start, end + offset, step)

    def walk__assigment(self, node):
        value = self.walk(node.value)
        self.scope.put(node.name, value)
        return value

    def walk__operation(self, node):
        if node.op not in ARIT_OPS:
            raise ValueError(f"Unknown operation: {node.op}")
        return ARIT_OPS[node.op](self.walk(node.left), self.walk(node.right))

    def walk__variable(self, node):
        value = self.scope.lookup(node.name)
        if value is None:
            raise ValueError(f"Variable '{node.name}' is undefined")
        return value

    def walk__integer(self, node):
        return int(node.value)


def in_opt_range(value: int,
                 ran: Optional[Tuple[Optional[int], Optional[int]]]) -> bool:
    if ran is None:
        return True

    left, right = ran

    return (left is None or left <= value) \
        and (right is None or right >= value)


def argument_check(number: Optional[Tuple[Optional[int],
                                          Optional[int]]] = None):
    def dec(func):
        def wrapper(ctx: MapperWalker, args: List[Any]):
            if not in_opt_range(len(args), number):
                raise RuntimeError("The number of args don't match function")
            func(ctx, args)
        return wrapper
    return dec


@argument_check(number=(1, None))
def map_print(_: MapperWalker, args: List[Any]):
    print(*args)


@argument_check(number=(1, 1))
def mov(mapper: MapperWalker, args: List[Any]):
    mapper.ship.move_fwd(args[0])


@argument_check(number=(1, 1))
def rot(mapper: MapperWalker, args: List[Any]):
    mapper.ship.rotate(args[0])


COLS = ['black', 'red', 'green', 'blue']


@argument_check(number=(1, 1))
def put(mapper: MapperWalker, args: List[Any]):
    mapper.ship.put(color=COLS[args[0]])


@argument_check(number=(1, 1))
def pen(mapper: MapperWalker, args: List[Any]):
    mapper.ship.pendown = bool(args[0])


@argument_check(number=(1, 1))
def col(mapper: MapperWalker, args: List[Any]):
    mapper.ship.color = COLS[args[0]]


INBUILT = {
    'print': map_print,
    'mov': mov,
    'rot': rot,
    'put': put,
    'pen': pen,
    'col': col
}

Model = Any


def compile_code(code: str) -> Model:
    parser = mapper_parser(True)
    processed = mapper_preprocessor(code)
    return parser.parse(processed)


def run_models(models: List[Model]) -> Tuple[Any, Map, Ship]:
    mapper_map = Map(start=(0, 0))
    ship = Ship(mapper_map, mapper_map.start)

    scopestack = ScopeStack()
    scopestack.add_scope()
    walker = MapperWalker(scopestack, INBUILT, ship, mapper_map)
    for model in models:
        for dec in model['decls']:
            walker.walk(dec)
    scopestack.add_scope()
    result = walker.walk(scopestack.lookup('main').body)

    return result, mapper_map, ship


if __name__ == "__main__":

    with open("tut.mpp") as f:
        code = f.read()

    models = [compile_code(code)]
    result, mapp, ship = run_models(models)

    print(export_map(mapp))
    render_map("level_0.png", mapp, ship)
