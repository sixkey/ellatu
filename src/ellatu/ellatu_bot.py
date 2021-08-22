import logging
import re
from typing import Callable, Dict, List, Optional, Tuple, Union
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
# Message splitting and building
###############################################################################


ParsingType = str
Splitter = Callable[[str], List[Tuple[ParsingType, str]]]


class ParsingNode:
    def __init__(self, p_type: ParsingType, value: Union[str,
                                                         List['ParsingNode']]):
        self.p_type = p_type
        self.value = value

    def split(self, rules: Dict[ParsingType, Splitter]) -> None:
        if isinstance(self.value, str):
            if self.p_type not in rules:
                return
            self.value = [ParsingNode(t, v)
                          for t, v in rules[self.p_type](self.value)]
        for child in self.value:
            child.split(rules)

    def collect(self, res: List[str]) -> List[str]:
        if isinstance(self.value, str):
            res.append(self.value)
        else:
            for child in self.value:
                child.collect(res)
        return res


class ParsingTree:
    def __init__(self, root: Optional[ParsingNode]):
        self.root: Optional[ParsingNode] = root

    def split(self, rules: Dict[ParsingType, Splitter]) -> None:
        if self.root is None:
            return
        self.root.split(rules)

    def collect(self) -> List[str]:
        if self.root is None:
            return []
        return self.root.collect([])


Particle = Tuple[ParsingType, str]


def find_codeblocks(value: str) -> List[Particle]:
    words = value.split('```')
    return [('codeblock', '```' + word + '```')
            if index % 2 == 1 else ('text', word)
            for index, word in enumerate(words)]


def find_codebits(value: str) -> List[Particle]:
    words = value.split('`')
    return [('codebit' if index % 2 == 1 else 'text', word)
            for index, word in enumerate(words)]


def find_lines(value: str) -> List[Particle]:
    lines = value.splitlines()
    return [('atom', w + ('\n' if i < len(lines) - 1 else ''))
            for i, w in enumerate(lines)]


PARSING_RULES = {
    'paragraph': find_codeblocks,
    'rawtext': find_codebits,
    'text': find_lines
}


def get_atomic_message_parts(message: str) -> List[str]:
    tree = ParsingTree(ParsingNode("paragraph", message))
    tree.split(PARSING_RULES)
    return tree.collect()


class DisMSegment:

    def __init__(self, max_segment_size: int, title: Optional[str] = None,
                 text_blocks: Optional[List[str]] = None):
        self.title = title
        self.text_blocks = text_blocks if text_blocks is not None else []
        self._size = sum([len(t) for t in self.text_blocks])
        self.max_segment_size = max_segment_size
        self.acc = ''

    def add(self, text_block: str, terminator: str = '') -> bool:
        if self.char_size() + len(text_block) > self.max_segment_size:
            return False
        self.text_blocks.append(text_block)
        self._size += len(text_block)
        self.acc += text_block + terminator
        return True

    def char_size(self) -> int:
        return (len(self.title) if self.title else 0) + self._size

    def value(self) -> str:
        return self.acc

    def __bool__(self) -> bool:
        return self.title is not None or bool(self.text_blocks)

    def __str__(self) -> str:
        return str(self.title) + ':\n' + self.value()


class DisMPage:

    def __init__(self, max_page_size: int):
        self.segments: List[DisMSegment] = []
        self._size = 0
        self.max_page_size = max_page_size
        self.images = []

    def add_segment(self, segment: DisMSegment) -> bool:
        if self.char_size() + segment.char_size() > self.max_page_size:
            return False
        self.segments.append(segment)
        self._size += segment.char_size()
        return True

    def char_size(self) -> int:
        return self._size

    def add_image(self, image: Tuple[str, str]):
        self.images.append(image)

    def __bool__(self) -> bool:
        return bool(self.segments) or bool(self.images)

    def __str__(self) -> str:
        return "\n\n".join([str(s) for s in self.segments])


class DisMBuilder:

    def __init__(self, max_segment_size: int = 1000,
                 max_page_size: int = 6000):
        self.pages: List[DisMPage] = []

        self.cur_page: DisMPage = DisMPage(max_page_size)
        self.cur_segment: DisMSegment = DisMSegment(max_segment_size)

        self.max_segment_size = max_segment_size
        self.max_page_size = max_page_size

    def flush_page(self) -> 'DisMBuilder':
        if not self.cur_page:
            return self
        self.pages.append(self.cur_page)
        self.cur_page = DisMPage(self.max_page_size)
        return self

    def flush_segment(self) -> 'DisMBuilder':
        if not self.cur_segment:
            return self
        if not self.cur_page.add_segment(self.cur_segment):
            self.flush_page()
            if not self.cur_page.add_segment(self.cur_segment):
                raise ValueError("Segment doesn't fit into the page")
        self.cur_segment = DisMSegment(self.max_segment_size)
        return self

    def set_title(self, title: str) -> 'DisMBuilder':
        self.cur_segment.title = title
        return self

    def add_text_block(self, text: str) -> 'DisMBuilder':
        if self.cur_segment.add(text.rstrip(), '\n\n'):
            return self
        atoms = get_atomic_message_parts(text)
        for index, atom in enumerate(atoms):
            term, value = ('', atom) \
                if index < len(atoms) - 1 else ('\n\n', atom.rstrip())
            if self.cur_segment.add(value, term):
                continue
            self.flush_segment()
            if not self.cur_segment.add(atom, term):
                raise ValueError("An text atom doesn't fit in empty segment")
        return self

    def add_image(self, image: Tuple[str, str]) -> 'DisMBuilder':
        self.cur_page.add_image(image)
        return self


###############################################################################
# Event
###############################################################################


def create_embed(title: str, color: discord.Color, desc: Optional[str] = None)\
        -> discord.Embed:
    if desc is None:
        return discord.Embed(title=title, color=color)
    return discord.Embed(title=title, color=color, description=desc)


def add_field(embed: discord.Embed, title: Optional[str] = None,
              value: Optional[str] = None, inline: bool = False) -> None:
    embed.add_field(
        name=title if title is not None else '\u200b',
        value=value if value is not None and value.strip() != '' else '-',
        inline=inline
    )


def build_pages(request: Request) -> DisMBuilder:
    builder = DisMBuilder()
    for message in request.messages:
        if isinstance(message, TextMessage):
            builder.add_text_block(message.message)
        elif isinstance(message, ParagraphMessage):
            builder.add_text_block(message.message)
            for image in message.images:
                builder.add_image(image)
        elif isinstance(message, MessageSegment):
            builder.flush_segment().set_title(message.title)
        elif isinstance(message, ImageMessage):
            builder.add_image((message.alt_text, message.location))
        else:
            builder.add_text_block(str(message))
    builder.flush_segment()
    builder.flush_page()
    return builder


async def send_response(request: Request, channel: TextChannel,
                        title: str = "Response",
                        inline: bool = True,
                        desc: Optional[str] = None) -> None:
    color = discord.Color.green() if request.alive else discord.Color.red()
    builder = build_pages(request)
    pages_num = len(builder.pages)

    for index, page in enumerate(builder.pages):
        page_embed = create_embed(title, color,
                                  desc=None if index > 0 else desc)
        for segment in page.segments:
            add_field(page_embed, title=segment.title,
                      value=segment.value(), inline=inline)

        image_file = None

        if page.images:
            _, thumb_file = page.images[0]
            image_file = discord.File(thumb_file, "thumb.png")
            page_embed.set_image(url="attachment://thumb.png")

        if pages_num > 1:
            page_embed.set_footer(text=f"{index + 1}/{pages_num}")

        await channel.send(embed=page_embed, file=image_file)

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


class TriggerCheckFailed(commands.CheckFailure):
    pass


def check_trigger(trigger_event):
    if isinstance(trigger_event, str):
        def pred_str(ctx):
            if not (hasattr(ctx, 'trigger_event')
                    and ctx.trigger_event == trigger_event):
                raise TriggerCheckFailed("Invalid trigger")
            return True
        return pred_str
    elif isinstance(trigger_event, set):
        def pred_set(ctx):
            if not (hasattr(ctx, 'trigger_event')
                    and ctx.trigger_event in trigger_event):
                raise TriggerCheckFailed("Invalid trigger")
            return True
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
        if isinstance(error,
                      (commands.CommandNotFound,
                       commands.CommandOnCooldown,
                       commands.UserInputError,
                       commands.MissingPermissions)):
            await ctx.send(f"{str(error)}")
        elif isinstance(error, TriggerCheckFailed):
            pass
        else:
            await ctx.send("Oops, internal error occured, contact admin")
            if isinstance(error,
                          commands.CheckFailure):
                ellatu_logger.log(logging.INFO, error)
            else:
                ellatu_logger.exception(error)
