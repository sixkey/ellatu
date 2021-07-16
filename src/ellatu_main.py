from typing import Dict, Optional, List
from ellatu import terminate_request
from mapper_pipeline import MapperPipeline
from ellatu_db import EllatuDB, Document
from ellatu import Ellatu, Request, RequestAction
import os
import logging
from ellatu_bot import EllatuCommandCog, EllatuListeningCog
from dotenv import load_dotenv
from mapper import mapper_parser
import json

from discord.ext import commands

def pipeline_tree(tree: Dict[str, RequestAction]) -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request, "No level set")
        if request.level['pipeline'] not in tree:
            return terminate_request(request, "Unknown pipeline")
        return tree[request.level['pipeline']](request)
    return action


class JSONEditor:

    def __init__(self, json: Optional[Document] = None):
        self._doc = json if json is not None else {}

    def inject(self, doc: Document) -> 'JSONEditor':
        for key, value in doc.items():
            self._doc[key] = value
        return self

    def select(self, keys: List[str]) -> 'JSONEditor':
        new_doc = {}
        for key in keys:
            if key in self._doc:
                new_doc[key] = self._doc[key]
        self._doc = new_doc
        return self

    def relabel(self, relabling: Dict[str, str]) -> 'JSONEditor':
        for oldkey, newkey in relabling.items():
            self._doc[newkey] = self._doc[oldkey]
            del self._doc[oldkey]
        return self

    def mat(self) -> Document:
        return self._doc


def loadworld(ellatudb: EllatuDB, path: str) -> None:
    with open(path) as f:
        world = json.load(f)
        world_doc = JSONEditor(world).select(
            ['title', 'code', 'tags', 'prereq']).mat()
        ellatudb.world.d_update(['code'], world_doc)
        for level in world['levels']:
            level_doc = JSONEditor(level).select(
                ['title', 'desc', 'code', 'tags', 'prepeq', 'tests', 'pipeline']
            ).inject(
                {'worldcode': world['code']}
            ).mat()
            ellatudb.level.d_update(['code', 'worldcode'], level_doc)




if __name__ == "__main__":

    # set up logging
    logger = logging.getLogger('discord')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(handler)

    # load enviroment variables
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    GUILD = os.getenv('DISCORD_GUILD')

    ellatudb = EllatuDB("localhost", 27017)
    loadworld(ellatudb, 'mapper.json')

    ellatu = Ellatu(ellatudb)

    mapper_pipeline = MapperPipeline()
    ellatu.on_submit_workflow = pipeline_tree({
        "mapper": mapper_pipeline.on_submit()
    })
    ellatu.on_run_workflow = pipeline_tree({
        "mapper": mapper_pipeline.on_run()
    })

    ellatu_bot = commands.Bot(command_prefix='!')
    ellatu_bot.add_cog(EllatuCommandCog(ellatu_bot, ellatu))
    ellatu_bot.add_cog(EllatuListeningCog(ellatu_bot, ellatu))
    ellatu_bot.run(TOKEN)
