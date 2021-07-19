import pygraphviz as pgv
import image_editing as imge

dot = pgv.AGraph(strict=False, directed=True)
dot.node_attr["shape"] = "rect"
dot.node_attr["style"] = "filled"


fillcolor = "white"
dot.add_node('a', fillcolor=fillcolor)
fillcolor = "#56de3e"
dot.add_node('b', fillcolor=fillcolor)
fillcolor = "#d63c31"
dot.add_node('c', fillcolor=fillcolor)
fillcolor = "#5c70d6"
dot.add_node('d', fillcolor=fillcolor)

dot.add_edge('a', 'b')
dot.add_edge('b', 'c')
dot.add_edge('a', 'd')
dot.layout(prog='dot')

filename = 'dot.png'
dot.draw(filename, format='png')

imge.edit_image(filename, imge.edit_sequence([
    imge.expand_to_aspect(1),
    imge.expand(total=20)
]))
