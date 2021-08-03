from typing import Dict, Optional, List

from .ellatu import pipeline_tree, Ellatu
from .ellatu_bot import EllatuCommandCog, EllatuListeningCog
from .ellatu_db import EllatuDB, Document
from .mapper_pipeline import MapperPipeline, get_level_settings

import os
import logging
from dotenv import load_dotenv
import json

from discord.ext import commands




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
            ['title', 'code', 'tags', 'prereqs']).mat()
        ellatudb.world.d_update(['code'], world_doc)
        for levelcode in world['levels']:
            with open(os.path.join('mapper', f"{levelcode}.json"), 'r') as lvl:
                level = json.load(lvl)
                level_doc = JSONEditor(level).select(
                    ['title', 'desc', 'code', 'tags', 'prereqs',
                     'tests', 'pipeline', 'attrs']
                ).inject(
                    {'worldcode': world['code']}
                ).mat()
                ellatudb.level.d_update(['code', 'worldcode'], level_doc)


def mapper_header(level: Document) -> Optional[str]:
    if level['pipeline'] == 'mapper':
        settings = get_level_settings(level)
        return "*max users: {}*\n".format(settings['users'] \
                                        if settings['users'] is not None \
                                        else 'unlimited') + \
               f"*max blocks: {settings['blocks']}*\n" + \
               f"*max lines: {settings['lines']}*\n" + \
               f"*max cols: {settings['cols']}*\n"
    return None


if __name__ == "__main__":

    # set up logging
    logger = logging.getLogger('discord')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(
        filename='discord.log', encoding='utf-8', mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(handler)

    # load enviroment variables
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    GUILD = os.getenv('DISCORD_GUILD')

    ellatudb = EllatuDB("localhost", 27017)



    loadworld(ellatudb, 'mapper/mapper.json')

    ellatu = Ellatu(ellatudb)

    mapper_pipeline = MapperPipeline()

    ellatu.on_submit_workflow = pipeline_tree({
        "mapper": mapper_pipeline.on_submit()
    })

    ellatu.on_run_workflow = pipeline_tree({
        "mapper": mapper_pipeline.on_run()
    })

    ellatu.header = mapper_header

    ellatu_bot = commands.Bot(command_prefix='!')
    ellatu_bot.add_cog(EllatuCommandCog(ellatu_bot, ellatu))
    ellatu_bot.add_cog(EllatuListeningCog(ellatu_bot, ellatu))
    ellatu_bot.run(TOKEN)
