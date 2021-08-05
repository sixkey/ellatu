from collections import deque
import logging
import re
from typing import List, Optional, Tuple
import discord
from discord.channel import TextChannel
from discord.errors import InvalidArgument
from discord.ext import commands

from .ellatu import (Ellatu, ImageMessage, Request, TextMessage,
                     MessageSegment, ParagraphMessage, pipeline_sequence)
from .ellatu_db import UserKey

ellatu_logger = logging.getLogger('ellatu')

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
            raise RuntimeError(f"A line can't be more than {message_size} " +
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


def add_field(embed: discord.Embed, title: Optional[str] = None,
              value: Optional[str] = None, inline: bool = False) -> None:
    embed.add_field(
        name=title if title is not None else '\u200b',
        value=value if value is not None and value.strip() != '' else '-',
        inline=inline
    )


def flush_blocks(embed: discord.Embed, title: Optional[str],
                 blocks: List[str], inline: bool = True) -> None:
    if not title and not blocks:
        return

    queue = deque(blocks)
    while queue:
        value = ''
        while queue and len(value) + len(queue[0]) < 1000:
            value += '\n\n' + queue.popleft()
        if title or value:
            add_field(embed, title=title, value=value, inline=inline)
        title = None
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
    request.ellatu.run_request(request.on_resolved(), request)


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

def check_trigger(trigger_event):
    if isinstance(trigger_event, str):
        def pred_str(ctx):
            return hasattr(ctx, 'trigger_event') \
                and ctx.trigger_event == trigger_event
        return pred_str
    elif isinstance(trigger_event, set):
        def pred_set(ctx):
            return hasattr(ctx, 'trigger_event') \
                and ctx.trigger_event in trigger_event
        return pred_set
    raise InvalidArgument("Trigger event can only be str or set")


class EllatuCommandCog(commands.Cog):
    def __init__(self, bot, ellatu):
        self.bot = bot
        self.ellatu = ellatu
        self.event_mode = None

    @commands.command(
        aliases=['hi'],
        help="Sign up for the bot (needs to be done only once, or after a " +
        "name change).",
        brief="sign up"
    )
    @commands.check(check_trigger('on_message'))
    async def hello(self, ctx) -> None:
        self.ellatu.user_connected(dc_userkey(ctx.author), ctx.author.name)
        await ctx.send(f'Hello {ctx.author.name}')

    @commands.command(
        aliases=['lvls'],
        help="Lists levels for current or selected world.",
        brief="lists levels"
    )
    @commands.check(check_trigger('on_message'))
    async def levels(self, ctx, worldcode: Optional[str] = None) -> None:
        request = self.ellatu.run_new_request(
            self.ellatu.get_levels(dc_userkey(ctx.author), worldcode),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Levels")

    @commands.command(
        help="Lists all worlds.",
        brief="lists worlds"
    )
    @commands.check(check_trigger('on_message'))
    async def worlds(self, ctx) -> None:
        request = self.ellatu.run_new_request(
            self.ellatu.get_worlds(),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Worlds")

    @commands.command(
        aliases=['mov', 'mv'],
        help="Move to a level written as '<worldcode>-<levelcode>' or only " +
        "'<levelcode>' in the current world. Example: " +
        "'!move world-level' or '!move level'.",
        brief="move to level"
    )
    @commands.check(check_trigger('on_message'))
    async def move(self, ctx, levelcode: str) -> None:
        request = self.ellatu.run_new_request(
            self.ellatu.user_move(dc_userkey(ctx.author), levelcode),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Move", inline=False)

    @commands.command(
        help="Displays information about the current level.",
        brief="info about level"
    )
    @commands.check(check_trigger('on_message'))
    async def sign(self, ctx) -> None:
        request = self.ellatu.run_new_request(
            self.ellatu.sign_for_user(dc_userkey(ctx.author)),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Sign", inline=False)

    @commands.command(
        help="Create workbench with codeblocks, after empty !submit, blocks " +
        "in these codeblocks are submitted. This command wors with " +
        "message edit, meaning you can edit code here and only use " +
        "!submit and !run. (workbenches may not work after time).",
        brief="create workbench"
    )
    async def workbench(self, ctx, *, text: str) -> None:
        codeblocks = extract_code_blocks(text)
        request = self.ellatu.run_new_request(
            self.ellatu.workbench(dc_userkey(ctx.author), codeblocks),
            dc_userkey(ctx.author)
        )
        if request.alive:
            await ctx.send(f"Workbench saved for {ctx.author.name}")
        else:
            await send_response(request, ctx.channel, title="Workbench")

    @commands.command(
        aliases=['s'],
        help="Submit codeblocks (blocks created using triple `), these " +
        "codeblocks are only saved, they still need to be run. If there " +
        "are no blocks in the message, blocks from !workbench are used.",
        brief="submit codeblocks"
    )
    @commands.check(check_trigger('on_message'))
    async def submit(self, ctx, *, text: Optional[str] = None) -> None:
        codeblocks = extract_code_blocks(text) if text is not None else None
        request = self.ellatu.run_new_request(
            self.ellatu.submit(dc_userkey(ctx.author), codeblocks),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Submit")

    @commands.command(
        aliases=['r'],
        help="Run submitted codeblocks against tests from current level, " +
        "tag other users to submit their codeblocks. (This feature may " +
        "or may not be allowed, depending on the level.",
        brief="run submitted blocks"
    )
    @commands.check(check_trigger('on_message'))
    async def run(self, ctx, users: commands.Greedy[discord.Member]) -> None:
        userkeys = [dc_userkey(u) for u in reversed(users)] + \
            [dc_userkey(ctx.author)]
        request = self.ellatu.run_new_request(
            self.ellatu.run(userkeys),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Run")

    @commands.command(
        help="Submit and run combined, the fact that run failed doesn't mean" +
        " that submit failed also (no rollback).",
        brief="submit and run"
    )
    @commands.check(check_trigger('on_message'))
    async def subrun(self, ctx, users: commands.Greedy[discord.Member], *,
                     text: Optional[str] = None) -> None:
        codeblocks = extract_code_blocks(text) if text is not None else None
        userkeys = [dc_userkey(ctx.message.author)] + \
            [dc_userkey(u) for u in users]
        request = self.ellatu.run_new_request(
            pipeline_sequence([
                self.ellatu.submit(dc_userkey(ctx.author), codeblocks),
                self.ellatu.run(userkeys)
            ]),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Submit and Run")

    @commands.command(
        help="Draw map of the current or selected world.",
        brief="draw map"
    )
    @commands.check(check_trigger('on_message'))
    async def map(self, ctx, worldcode=None) -> None:
        request = self.ellatu.run_new_request(
            self.ellatu.draw_map(dc_userkey(ctx.message.author), worldcode),
            dc_userkey(ctx.author)
        )
        await send_response(request, ctx.channel, title="Map")

    # Listeners ###

    async def process_commands(self, message, trigger_event) -> None:
        if message.author.bot:
            return
        ctx = await self.bot.get_context(message)
        ctx.trigger_event = trigger_event
        await self.bot.invoke(ctx)

    @commands.Cog.listener()
    async def on_message(self, message) -> None:
        await self.process_commands(message, "on_message")

    @commands.Cog.listener()
    async def on_message_edit(self, _, message) -> None:
        await self.process_commands(message, "on_message_edit")

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        await ctx.send(f"{str(error)}")
