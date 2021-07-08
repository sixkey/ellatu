from discord.channel import TextChannel
from ellatu import Ellatu, Request
import discord
import re

from typing import List, Optional

def extract_code_blocks(message: str) -> List[str]:
    result: List[str] = []
    for group in re.findall(r'```([^`]*)```',message):
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

class EllatuBot(discord.Client):

    def set_ellatu(self, ellatu: Ellatu) -> None:
        self.ellatu = ellatu

    async def on_ready(self):
        print(f'{self.user}')

    async def send_response(self, request: Request, channel: TextChannel) -> None:
        embed = discord.Embed(title="Response")
        embed.add_field(name="response", value=str(request))
        await channel.send(embed=embed)

    async def on_message(self, message):
        print(f'{message.author}, {message.content}')

        if message.author == self.user or not is_command(message.content):
            return

        command = get_command(message.content)
        if command is None:
            return

        if command == 'submit':
            codeblocks = extract_code_blocks(message.content)
            print(codeblocks)
            request = self.ellatu.submit(str(message.author.id), codeblocks)
            print(str(request))
            await self.send_response(request, message.channel)


        elif command == 'hello':
            self.ellatu.user_connected(str(message.author.id))

        elif command == 'level':
            pass

        elif command == 'move':
            pass
            self.ellatu.user_move(str(message.author.id))
