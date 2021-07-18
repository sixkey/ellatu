from inspect import Traceback
from typing import List
from ellatu import Request, data_action, limit_codeblocks, limit_columns, \
                  limit_lines, terminate_request, trace, pipeline_sequence, \
                  RequestAction, EllatuPipeline, MessageType, ParagraphMessage
import mapper

PARSER = mapper.mapper_parser()


def compile_codeblocks(parser) -> RequestAction:
    def action(request: Request) -> Request:
        models = []

        for _, codeblocks in request.codeblocks.items():
            try:
                models += mapper.compile_codeblocks(codeblocks, parser)
            except Exception as e:

                return terminate_request(request, "Compilation error:\n " +
                                         str(e))

        request.data["models"] = models
        return trace(request, "Compilation success")
    return action


@data_action(["models"])
def run_models(request: Request, models: List[mapper.Model]) -> Request:
    if request.level is None:
        return terminate_request(request, "Invalid level")

    _, mapp, ship = mapper.run_models(models)

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
        return pipeline_sequence([
            limit_codeblocks(3),
            limit_lines(10),
            limit_columns(79),
            compile_codeblocks(self.parser)
        ])

    def on_run(self) -> RequestAction:
        return pipeline_sequence([
            self.on_submit(),
            run_models
        ])
