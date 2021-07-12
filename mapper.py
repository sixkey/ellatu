from typing import Any, Callable, List, Optional, Tuple, Dict, TypeVar
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from math import pi
from pprint import pprint
import tatsu
from tatsu.walkers import NodeWalker
import json

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
# RENDERING
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
# PARSING
###############################################################################

def mapper_parser(asmodel = True):
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
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a // b,
    '%': lambda a, b: a % b
}


class MapperWalker(NodeWalker):

    def __init__(self, glob_scope: ScopeStack):
        self.ctx = glob_scope
        self.scope = glob_scope

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

    def walk__lift(self, node):
        return (node.lift, self.walk(node.value))

    def walk__amnesia_stmt(self, node):
        self.walk(node.stmt)
        return None

    def walk__call(self, node):
        function = self.scope.lookup(node.name)
        if function is None:
            raise ValueError(f"function is not defined")

        if len(function.params) != (len(node.args)):
            raise ValueError(f"Invalid number of positional arguments")

        scope = self.scope
        newscope = self.scope.copy()

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


if __name__ == "__main__":

    mapper_map = fill_from_file(Map(), "map.txt")
    ship = Ship(mapper_map.start)

    parser = mapper_parser(True)

    with open("main.mapper") as f:
        code = f.read()
        code = mapper_preprocessor(code)
        print(code)
        model = parser.parse(code)

        scopestack = ScopeStack()
        scopestack.add_scope()

        walker = MapperWalker(scopestack)
        for dec in model['decls']:
            walker.walk(dec)
        scopestack.add_scope()
        result = walker.walk(scopestack.lookup('main').body)
        print(None if result is None else result[1])

        # pprint(model)

    # render_map("image.png", mapper_map, ship)
