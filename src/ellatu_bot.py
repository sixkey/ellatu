import re
from ellatu_db import UserKey
from ellatu import Ellatu, ParagraphMessage
from typing import List, Optional
import discord
from discord.channel import TextChannel
from discord.ext import commands
from ellatu import Message, Request, TextMessage

###############################################################################
# Utils
###############################################################################


def dc_userkey(user: discord.User) -> UserKey:
    return ('dc', str(user.id))


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

###############################################################################
# Event
###############################################################################


def create_embed(title: str, color: discord.Color, desc: Optional[str] = None)\
    -> discord.Embed:
    if desc is None:
        return discord.Embed(title=title, color=color)
    return discord.Embed(title=title, color=color, description=desc)


def embed_message(embed: discord.Embed, message: Message) -> Optional[discord.File]:
    if isinstance(message, TextMessage):
        embed.add_field(name="\u200b", value=message.message,
                        inline=False)
    elif isinstance(message, ParagraphMessage):
        embed.add_field(name="\u200b", value=message.message,
                        inline=False)
        if message.images:
            _, thumb_file = message.images[0]
            print(thumb_file)
            f = discord.File(thumb_file, "thumb.png")
            embed.set_image(url="attachment://thumb.png")
            return f
    else:
        embed.add_field(name="\u200b", value=str(message),
                        inline=False)
    return None


async def send_response(request: Request, channel: TextChannel,
                        title: str = "Response",
                        desc: Optional[str] = None) -> None:
    color = discord.Color.green() if request.alive else discord.Color.red()
    embed = create_embed(title, color, desc)

    image_file = None
    for message in request.messages:
        image_file = embed_message(embed, message)
    await channel.send(embed=embed, file=image_file)


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
    async def levels(self, ctx, worldcode: str) -> None:
        request = self.ellatu.get_levels(dc_userkey(ctx.author), worldcode)
        await send_response(request, ctx.channel, title="Levels",
                            desc=f"levels in **{worldcode}**")

    @commands.command()
    async def worlds(self, ctx) -> None:
        request = self.ellatu.get_worlds()
        await send_response(request, ctx.channel)

    @commands.command()
    async def move(self, ctx, levelcode: str) -> None:
        request = self.ellatu.user_move(dc_userkey(ctx.author), levelcode)
        await send_response(request, ctx.channel)

    @commands.command()
    async def sign(self, ctx) -> None:
        request = self.ellatu.sign_for_user(dc_userkey(ctx.author))
        await send_response(request, ctx.channel)

    @commands.command()
    async def submit(self, ctx, *, text: str) -> None:
        codeblocks = extract_code_blocks(text)
        request = self.ellatu.submit(dc_userkey(ctx.author), codeblocks)
        await send_response(request, ctx.channel)

    @commands.command()
    async def run(self, ctx, users: commands.Greedy[discord.Member]) -> None:
        userkeys = [dc_userkey(ctx.message.author)] + \
            [dc_userkey(u) for u in users]
        request = self.ellatu.run(userkeys)
        await send_response(request, ctx.channel)
