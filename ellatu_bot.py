from discord.ext.commands.converter import Greedy
from discord.member import Member
from ellatu_db import EllatuDB
from discord.channel import TextChannel
from ellatu import Ellatu, Request
import discord
from discord.ext import commands
import re
import os
import logging
from dotenv import load_dotenv


from typing import List, Optional


def extract_code_blocks(message: str) -> List[str]:
    result: List[str] = []
    for group in re.findall(r'```([^`]*)```', message):
        result.append(group)
    return result


def starts_in(message: str, character: str) -> bool:
    if len(message) == 0:
        return False
    return message[0] == character


def is_command(message: str) -> bool:
    return starts_in(message.strip(), '!')


def get_command(message: str) -> Optional[str]:
    match = re.match(r"^\s*!(\w+)[\s\S]*", message)
    if match is None:
        return None

    return match.group(1)


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

ellatudb = EllatuDB("localhost", 27017)
ellatu = Ellatu(ellatudb)
ellatu_bot = commands.Bot(command_prefix='!')

###############################################################################
# EVENT
###############################################################################


@ellatu_bot.event
async def on_ready():
    print(f'{ellatu_bot.user}')


async def send_response(request: Request, channel: TextChannel) -> None:
    embed = discord.Embed(title="Response")
    embed.add_field(name="response", value=str(request))
    await channel.send(embed=embed)


async def send_error(channel: TextChannel, title: str,
                     message: Optional[str]) -> None:
    embed = discord.Embed(title=f"Oops: {title}")
    if message is not None:
        embed.add_field(name=None, value=message)
    await channel.send(embed=embed)


@ellatu_bot.event
async def on_message(message):
    print(f'{message.author}, {message.content}')

    if message.author == ellatu_bot.user or not is_command(message.content):
        return

    await ellatu_bot.process_commands(message)

###############################################################################
# COMMANDS
###############################################################################


@ellatu_bot.command()
async def hello(ctx):
    ellatu.user_connected(str(ctx.author.id))
    await ctx.send('Hello')


@ellatu_bot.command()
async def levels(ctx, worldcode: str):
    request = ellatu.get_levels(worldcode)
    await send_response(request, ctx.channel)


@ellatu_bot.command()
async def worlds(ctx):
    request = ellatu.get_worlds()
    await send_response(request, ctx.channel)

@ellatu_bot.command()
async def move(ctx, levelcode: str):
    request = ellatu.user_move(str(ctx.author.id), levelcode)
    await send_response(request, ctx.channel)

@ellatu_bot.command()
async def sign(ctx):
    request = ellatu.sign_for_user(str(ctx.author.id))
    await send_response(request, ctx.channel)

@ellatu_bot.command()
async def submit(ctx, *, text: str):
    codeblocks = extract_code_blocks(text)
    print(codeblocks)
    request = ellatu.submit(str(ctx.author.id), codeblocks)
    await send_response(request, ctx.channel)

@ellatu_bot.command()
async def run(ctx, users: commands.Greedy[discord.Member]):
    usernames = [str(u.id) for u in users]
    request = ellatu.run(usernames)
    await send_response(request, ctx.channel)



# set up logging
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(
    filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# load enviroment variables
ellatu_bot.run(TOKEN)
