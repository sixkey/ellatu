from ellatu_db import EllatuDB
from ellatu import Ellatu
import os
import logging
from ellatu_bot import EllatuBot
from dotenv import load_dotenv

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

    client = EllatuBot()
    client.set_ellatu(ellatu)
    client.run(TOKEN)
