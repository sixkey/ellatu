from collections import deque
import re
from typing import List, Optional, Tuple
import discord
from discord.channel import TextChannel
from discord.ext import commands

from .ellatu import Ellatu, ImageMessage, Request, TextMessage, MessageSegment, \
    ParagraphMessage
from .ellatu_db import UserKey

###############################################################################
# Utils
###############################################################################


def dc_userkey(user: discord.User) -> UserKey:
    return ('dc', str(user.id))


def extract_code_blocks(message: str) -> List[str]:
    result: List[str] = []
    for group in re.findall(r'```[^\n]*([^`]*)```', message):
        result.append(group)
    return result


def starts_in(message: str, character: str) -> bool:
    if len(message) == 0:
        return False
    return message[0] == character


def is_command(message: str) -> bool:
    return starts_in(message.strip(), '!')

###############################################################################
# Event
###############################################################################


def create_embed(title: str, color: discord.Color, desc: Optional[str] = None)\
        -> discord.Embed:
    if desc is None:
        return discord.Embed(title=title, color=color)
    return discord.Embed(title=title, color=color, description=desc)

def split_text(text: str, message_size: int) -> List[str]:
    chunks = []

    # This is a very naive solution with many problems, but will have to do now

    lines = text.split('\n')
    buf = ''
    for line in lines:
        if len(line) > message_size:
            raise RuntimeError(f"A line can't be more than {message_size} " + \
                               "characters long")
        if len(buf) + len(line) + 1 > message_size:
            chunks.append(buf)
            buf = ''
        if buf != '':
            buf += '\n'
        buf += line
    if buf != '':
        chunks.append(buf)
    return chunks

def add_block(blocks: List[str], block: str) -> None:
    if len(block) <= 1000:
        blocks.append(block)
        return

    for chunk in split_text(block, 1000):
        blocks.append(chunk)

def flush_blocks(embed: discord.Embed, title: Optional[str],
                 blocks: List[str], inline: bool = True) -> None:
    if not title and not blocks:
        return

    queue = deque(blocks)
    title = title if title else '\u200b'
    while queue:
        value = ''
        while queue and len(value) + len(queue[0]) < 1000:
            print("BLOCK")
            print(queue[0])
            value += '\n\n' + queue.popleft()
        embed.add_field(name=title, value=value, inline=inline)
        title = '\u200b'
    blocks.clear()

async def send_response(request: Request, channel: TextChannel,
                        title: str = "Response",
                        inline: bool = True,
                        desc: Optional[str] = None) -> None:
    color = discord.Color.green() if request.alive else discord.Color.red()
    embed = create_embed(title, color, desc)

    image_file = None

    text_blocks: List[str] = []
    images: List[Tuple[str, str]] = []
    name = None

    for message in request.messages:
        if isinstance(message, TextMessage):
            add_block(text_blocks, message.message)
        elif isinstance(message, ParagraphMessage):
            add_block(text_blocks, message.message)
            if message.images:
                images += message.images
        elif isinstance(message, MessageSegment):
            flush_blocks(embed, name, text_blocks, inline=inline)
            name = message.title
        elif isinstance(message, ImageMessage):
            images.append((message.alt_text, message.location))
        else:
            add_block(text_blocks, str(message))

    flush_blocks(embed, name, text_blocks, inline=inline)

    image_file = None
    if images:
        _, thumb_file = images[0]
        image_file = discord.File(thumb_file, "thumb.png")
        embed.set_image(url="attachment://thumb.png")
    await channel.send(embed=embed, file=image_file)
    request.on_resolved()


async def send_error(channel: TextChannel, title: str,
                     message: Optional[str]) -> None:
    embed = discord.Embed(title=f"Oops: {title}")
    if message is not None:
        embed.add_field(name=None, value=message)
    await channel.send(embed=embed)


###############################################################################
# Listening
###############################################################################

class EllatuListeningCog(commands.Cog):
    def __init__(self, bot: commands.Bot, ellatu: Ellatu) -> None:
        self.bot = bot
        self.ellatu = ellatu

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        print(f'{self.bot.user}')

    @commands.Cog.listener()
    async def on_message(self, message) -> None:
        print(f'{message.author}, {message.content}')

###############################################################################
# Commands
###############################################################################


class EllatuCommandCog(commands.Cog):
    def __init__(self, bot, ellatu):
        self.bot = bot
        self.ellatu = ellatu

    @commands.command()
    async def hello(self, ctx) -> None:
        self.ellatu.user_connected(dc_userkey(ctx.author), ctx.author.name)
        await ctx.send(f'Hello {ctx.author.name}')

    @commands.command()
    async def levels(self, ctx, worldcode: Optional[str] = None) -> None:
        request = self.ellatu.get_levels(dc_userkey(ctx.author), worldcode)
        await send_response(request, ctx.channel, title="Levels")

    @commands.command()
    async def worlds(self, ctx) -> None:
        request = self.ellatu.get_worlds()
        await send_response(request, ctx.channel, title="Worlds")

    @commands.command()
    async def move(self, ctx, levelcode: str) -> None:
        request = self.ellatu.user_move(dc_userkey(ctx.author), levelcode)
        await send_response(request, ctx.channel, title="Move", inline=False)

    @commands.command()
    async def sign(self, ctx) -> None:
        request = self.ellatu.sign_for_user(dc_userkey(ctx.author))
        await send_response(request, ctx.channel, title="Sign", inline=False)

    @commands.command()
    async def submit(self, ctx, *, text: str) -> None:
        codeblocks = extract_code_blocks(text)
        request = self.ellatu.submit(dc_userkey(ctx.author), codeblocks)
        await send_response(request, ctx.channel, title="Submit")

    @commands.command()
    async def run(self, ctx, users: commands.Greedy[discord.Member]) -> None:
        userkeys = [dc_userkey(ctx.message.author)] + \
            [dc_userkey(u) for u in users]
        request = self.ellatu.run(userkeys)
        await send_response(request, ctx.channel, title="Run")

    @commands.command()
    async def map(self, ctx, worldcode = None) -> None:
        request = self.ellatu.draw_map(dc_userkey(ctx.message.author), worldcode)
        await send_response(request, ctx.channel, title="Map")

#   @commands.Cog.listener()
#   async def on_command_error(self, ctx, error):
#       await ctx.send(f"{str(error)}")
