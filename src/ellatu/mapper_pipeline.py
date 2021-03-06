from .ellatu_db import Document
from typing import Dict, List, Optional
from .ellatu import MessageSegment, Request, add_msg, data_action, limit_codeblocks, limit_columns, \
    limit_lines, limit_users, remove_files, terminate_request, trace, pipeline_sequence, \
    RequestAction, EllatuPipeline, MessageType, ParagraphMessage
from . import mapper

PARSER = mapper.mapper_parser()


DEFAULT_SETTINGS: Dict[str, Optional[int]] = {
    "users": None,
    "blocks": 3,
    "lines": 10,
    "cols": 79
}

def compile_codeblocks(parser) -> RequestAction:
    def action(request: Request) -> Request:
        models = []

        for _, codeblocks in request.codeblocks.items():
            try:
                models += mapper.compile_codeblocks(codeblocks, parser)
            except mapper.MapperError as e:
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
    except mapper.MapperError as e:
        return terminate_request(request, "Mapper error: " + str(e))

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

def get_level_settings(level: Document) -> Dict[str, Optional[int]]:
    settings = dict(DEFAULT_SETTINGS)
    for key, value in level["attrs"].items():
        settings[key] = value
    return settings


def check_blocks(request: Request) -> Request:
    if request.level is None:
        return terminate_request(request, "The level is not set")

    settings = get_level_settings(request.level)
    return pipeline_sequence([
        limit_users(settings["users"]),
        limit_codeblocks(settings["blocks"]),
        limit_lines(settings["lines"]),
        limit_columns(settings["cols"])
    ])(request)

class MapperPipeline(EllatuPipeline):

    def __init__(self) -> None:
        self.parser = mapper.mapper_parser()

    def on_submit(self) -> RequestAction:
        return pipeline_sequence([
            check_blocks,
            add_msg(MessageSegment('Compilation')),
            compile_codeblocks(self.parser)
        ])

    def on_run(self) -> RequestAction:
        return pipeline_sequence([
            self.on_submit(),
            add_msg(MessageSegment('Testing')),
            run_models
        ])
