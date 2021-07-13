import re
from typing import List, Optional
import discord
from discord.channel import TextChannel
from discord.ext import commands
from ellatu import Message, Request, TextMessage

###############################################################################
# UTILS
###############################################################################


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
# EVENT
###############################################################################


def embed_message(embed, message: Message) -> None:
    if isinstance(message, TextMessage):
        embed.add_field(name="\u200b", value=message.message,
                        inline=False)
    else:
        embed.add_field(name="\u200b", value=str(message),
                        inline=False)


async def send_response(request: Request, channel: TextChannel,
                        title: str = "Response",
                        desc: Optional[str] = None) -> None:
    color = discord.Color.green() if request.alive else discord.Color.red()
    embed = discord.Embed(color=color, title=title,
                          description=desc)
    for message in request.messages:
        embed_message(embed, message)
    await channel.send(embed=embed)


async def send_error(channel: TextChannel, title: str,
                     message: Optional[str]) -> None:
    embed = discord.Embed(title=f"Oops: {title}")
    if message is not None:
        embed.add_field(name=None, value=message)
    await channel.send(embed=embed)


class EllatuListeningCog(commands.Cog):
    def __init__(self, bot, ellatu):
        self.bot = bot
        self.ellatu = ellatu

    @commands.Cog.listener()
    async def on_ready(self):
        print(f'{self.bot.user}')

    @commands.Cog.listener()
    async def on_message(self, message):
        print(f'{message.author}, {message.content}')

        if message.author == self.bot.user or not is_command(message.content):
            return


###############################################################################
# COMMANDS
###############################################################################


class EllatuCommandCog(commands.Cog):
    def __init__(self, bot, ellatu):
        self.bot = bot
        self.ellatu = ellatu

    @commands.command()
    async def hello(self, ctx):
        self.ellatu.user_connected(str(ctx.author.id))
        await ctx.send('Hello')

    @commands.command()
    async def levels(self, ctx, worldcode: str):
        request = self.ellatu.get_levels(worldcode)
        await send_response(request, ctx.channel, title="Levels",
                            desc=f"levels in **{worldcode}**")

    @commands.command()
    async def worlds(self, ctx):
        request = self.ellatu.get_worlds()
        await send_response(request, ctx.channel)

    @commands.command()
    async def move(self, ctx, levelcode: str):
        request = self.ellatu.user_move(str(ctx.author.id), levelcode)
        await send_response(request, ctx.channel)

    @commands.command()
    async def sign(self, ctx):
        request = self.ellatu.sign_for_user(str(ctx.author.id))
        await send_response(request, ctx.channel)

    @commands.command()
    async def submit(self, ctx, *, text: str):
        codeblocks = extract_code_blocks(text)
        request = self.ellatu.submit(str(ctx.author.id), codeblocks)
        await send_response(request, ctx.channel)

    @commands.command()
    async def run(self, ctx, users: commands.Greedy[discord.Member]):
        usernames = [str(u.id) for u in users]
        request = ellatu.run(usernames)
        await send_response(request, ctx.channel)
