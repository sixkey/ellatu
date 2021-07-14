from ellatu_db import EllatuDB
from ellatu import Ellatu
import os
import logging
from ellatu_bot import EllatuCommandCog, EllatuListeningCog
from dotenv import load_dotenv

from discord.ext import commands


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
    ellatu = Ellatu(ellatudb)

    ellatu_bot = commands.Bot(command_prefix='!')
    ellatu_bot.add_cog(EllatuCommandCog(ellatu_bot, ellatu))
    ellatu_bot.add_cog(EllatuListeningCog(ellatu_bot, ellatu))
    ellatu_bot.run(TOKEN)
