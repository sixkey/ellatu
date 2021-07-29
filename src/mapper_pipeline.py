from inspect import Traceback
from typing import List
from ellatu import MessageSegment, Request, add_msg, data_action, limit_codeblocks, limit_columns, \
    limit_lines, remove_files, terminate_request, trace, pipeline_sequence, \
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
        return trace(request, "Success")
    return action


@data_action(["models"])
def run_models(request: Request, models: List[mapper.Model]) -> Request:
    if request.level is None:
        return terminate_request(request, "Invalid level")
    try:
        _, mapp, ship = mapper.run_models(models)
    except RuntimeError as e:
        return terminate_request(request, "Runtime error: " + str(e))

    for test in request.level['tests']:
        print(test)
        desired_map = mapper.fill_from_text(mapper.Map(), test, ';')
        diff_map = mapper.mapp_diff(desired_map, mapp)
        if diff_map is not None:
            filename = request.ellatu.temp_files.add_temp_filename(
                str(request.id) + '-mapperdiff.png'
            )
            mapper.render_map(filename, diff_map, ship)
            request.add_message(
                ParagraphMessage(f"here is the diff ![diff][{filename}]",
                                 message_type=MessageType.ERROR))
            request.alive = False
            request.add_on_res(remove_files([filename]))
            return request

    return trace(request, "Passed all tests")


class MapperPipeline(EllatuPipeline):

    def __init__(self) -> None:
        self.parser = mapper.mapper_parser()

    def on_submit(self) -> RequestAction:
        return pipeline_sequence([
            limit_codeblocks(3),
            limit_lines(10),
            limit_columns(79),
            add_msg(MessageSegment('Compilation')),
            compile_codeblocks(self.parser)
        ])

    def on_run(self) -> RequestAction:
        return pipeline_sequence([
            self.on_submit(),
            add_msg(MessageSegment('Testing')),
            run_models
        ])
