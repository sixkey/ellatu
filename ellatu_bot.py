import discord
import re

from typing import List

def extract_code_blocks(message: str) -> List[str]:
    result: List[str] = []
    for group in re.findall(r'```([^`]*)```',message):
        result.append(group)
    return result

class EllatuBot(discord.Client):
    async def on_ready(self):
        print(f'{self.user}')

    async def on_message(self, message):
        print(f'{message.author}, {message.content}')
        code_blocks = extract_code_blocks(message.content)


