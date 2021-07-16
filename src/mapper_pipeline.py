from inspect import Traceback
from ellatu import Request, terminate_request, trace, sequence, RequestAction,\
                  EllatuPipeline, MessageType, ParagraphMessage
import mapper

PARSER = mapper.mapper_parser()


def limit_codeblocks(number: int) -> Request:
    def action(request: Request) -> Request:
        for _, codeblocks in request.codeblocks.items():
            if len(codeblocks) > number:
                return terminate_request(
                    request,
                    f"There were more then {number} lines"
                )
        return request
    return action


def limit_lines(number: int) -> Request:
    def action(request: Request) -> Request:
        for _, codeblocks in request.codeblocks.items():
            for codeblock in codeblocks:
                if len(codeblock.splitlines()) > number:
                    return terminate_request(
                        request,
                        f"A codeblock was longer than {number} lines"
                    )
        return request
    return action


def limit_columns(number: int) -> Request:
    def action(request: Request) -> Request:
        for _, codeblocks in request.codeblocks.items():
            for codeblock in codeblocks:
                for line in codeblock.splitlines():
                    if len(line) > number:
                        return terminate_request(
                            request,
                            f"Line was longer than {number} characters"
                        )
        return request
    return action


def compile_codeblocks(parser) -> RequestAction:
    def action(request: Request) -> Request:
        models = []

        for _, codeblocks in request.codeblocks.items():
            try:
                models += mapper.compile_codeblocks(codeblocks, parser)
            except Exception as e:

                return terminate_request(request, "Compilation error:\n " + 
                                         str(e))

        request.models = models
        return trace(request, "Compilation success")
    return action


def run_models(request: Request) -> Request:

    if request.models is None:
        return terminate_request(request, "There aren't any compiled blocks")
    if request.level is None:
        return terminate_request(request, "Invalid level")

    _, mapp, ship = mapper.run_models(request.models)

    for test in request.level['tests']:
        print(test)
        desired_map = mapper.fill_from_text(mapper.Map(), test, ';')
        diff_map = mapper.mapp_diff(mapp, desired_map)
        if diff_map is not None:
            mapper.render_map("temp.png", diff_map, ship)
            request.add_message(ParagraphMessage("here is the diff ![diff][temp.png]", message_type= MessageType.ERROR))
            request.alive = False
            return request 

    return trace(request, "The tests passed")


class MapperPipeline(EllatuPipeline):

    def __init__(self) -> None:
        self.parser = mapper.mapper_parser()

    def on_submit(self) -> RequestAction:
        return sequence([
            limit_codeblocks(3),
            limit_lines(10),
            limit_columns(79),
            compile_codeblocks(self.parser)
        ])

    def on_run(self) -> RequestAction:
        return sequence([
            self.on_submit(),
            run_models
        ])
