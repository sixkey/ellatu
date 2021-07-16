import re
from ellatu_db import Document, EllatuDB, MongoId, UserKey, get_userkey
from typing import Dict, List, Callable, Optional, Any
from enum import Enum


class MessageType(Enum):
    LOG = 1
    TRACE = 2
    ERROR = 3


class Message:

    def __init__(self, message_type: MessageType):
        self.message_type = message_type

    def prefix(self, message_body: str) -> str:
        return f"[{self.message_type}]: {message_body}"


class TextMessage(Message):

    def __init__(self, message: str,
                 message_type: MessageType = MessageType.LOG):
        super().__init__(message_type)
        self.message = message

    def __str__(self) -> str:
        return self.prefix(self.message)

class ParagraphMessage(Message):

    def __init__(self, message: str,
                 message_type: MessageType = MessageType.LOG):
        super().__init__(message_type)

        matches = re.split(r'!(\[[^\]]*\]\[[^\]]*\])',
                           message)
        images = []
        final_message = ""
        for index, part in enumerate(matches):
            if index % 2 == 0:
                final_message += part
            else:
                worlds = part[1:-1].split('][', maxsplit=1)
                images.append(worlds)
                final_message += f"_[{len(images)}]_"

        self.message = final_message
        self.images = images

    def __str__(self) -> str:
        return self.prefix(self.message)

class Request:

    def __init__(self, ellatu: "Ellatu",
                 codeblocks: Optional[Dict[MongoId, List[str]]] = None):
        self.ellatu = ellatu
        self.alive = True

        self.messages: List[Message] = []

        self.level: Optional[Document] = None
        self.levels: Dict[str, Document] = {}

        self.users: Dict[MongoId, Document] = {}

        self.codeblocks: Dict[MongoId, List[str]
                              ] = codeblocks if codeblocks else {}

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def __str__(self) -> str:
        res = f"Request: [{'Alive' if self.alive else 'Dead'}]\n"
        for message in self.messages:
            res += str(message) + '\n'
        return res


def log_msg(request: Request, message: str,
            message_type: MessageType = MessageType.LOG) -> Request:
    request.add_message(TextMessage(message, message_type=message_type))
    return request


def trace(request: Request, message: str) -> Request:
    return log_msg(request, message, MessageType.TRACE)


def log(request: Request, message: str) -> Request:
    return log_msg(request, message, MessageType.TRACE)


def error(request: Request, message: str) -> Request:
    return log_msg(request, message, MessageType.ERROR)


def terminate_request(request: Request, message: str) -> Request:
    request.alive = False
    return error(request, message)


RequestAction = Callable[[Request], Request]


def const_workflow(request: Request) -> Request:
    return request


def kill_request(request: Request) -> Request:
    return terminate_request(request, "The request has been killed")


def add_msg(message: Message) -> RequestAction:
    def action(request: Request) -> Request:
        request.add_message(message)
        return request
    return action


def add_users(userkeys: List[UserKey]) -> RequestAction:
    def action(request: Request) -> Request:
        keys_set = set(userkeys)
        users = request.ellatu.db.user.get_users(userkeys)
        for user in users:
            keys_set.remove(get_userkey(user))
            request.users[get_userkey(user)] = user
        if keys_set:
            terminate_request(request,
                              f"The users were not found")
        return request
    return action


def assign_codeblocks(assignments: Dict[UserKey, List[str]]) -> RequestAction:
    def action(request: Request) -> Request:
        for userkey, codeblocks in assignments.items():
            if userkey not in request.users:
                request.messages.append(TextMessage(
                    f"User not is not in the request"))
                continue
            user = request.users[userkey]
            request.codeblocks[user["_id"]] = codeblocks
        return request
    return action


def assign_from_workplace(userkeys: List[UserKey]) -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request,
                                     "The request doesn't have valid \
                                        level set")
        users = request.ellatu.db.user.get_users(userkeys)
        userids = [u["_id"] for u in users]
        codeblocks = request.ellatu.db.workplace\
            .get_codeblocks(userids, request.level["worldcode"])

        for userid, blocks in codeblocks.items():
            request.codeblocks[userid] = blocks
        return request
    return action

def print_codeblocks() -> RequestAction:
    def action(request: Request) -> Request:
        res = ""

        usermap = {}
        for userkey, user in request.users.items():
            usermap[user["_id"]] = userkey

        for userid, codeblocks in request.codeblocks.items():
            if userid not in usermap:
                return terminate_request(request, "Users not loaded corretly")

            user = request.users[usermap[userid]]
            res += user["username"]
            for codeblock in codeblocks:
                res += "```\n" + codeblock + "```"

        trace(request, res)

        return request
    return action


def localize_by_user(userkey: UserKey) -> RequestAction:
    def action(request: Request) -> Request:
        user = request.ellatu.db.user.get_user(userkey)
        if user is None:
            return terminate_request(request, "The user was not found")
        request.level = request.ellatu.db.level.get_by_code(user["levelcode"])
        return request
    return action


def localize_by_code(code: str) -> RequestAction:
    def action(request: Request) -> Request:
        match = re.match(r'^(\w+)\-(\w+)$', code)
        if match is None:
            return terminate_request(request, "Invalid level code format")
        worldcode, levelcode = match.group(1), match.group(2)
        level = request.ellatu.db.level.get_one(worldcode=worldcode, code=levelcode)
        if not level:
            return terminate_request(request, "Invalid level code")
        request.level = level
        return request
    return action


def move_users() -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request, "Invalid level")
        for userkey, _ in request.users.items():
            if not request.ellatu.db.user.move_user(userkey, request.level["worldcode"], request.level["code"]):
                return terminate_request(request, "Unable to move user")
        return request
    return action


def level_code_parts(worldcode: str, levelcode: str) -> str:
    return f"{worldcode}-{levelcode}"


def level_code_doc(level: Document) -> str:
    return level_code_parts(level["worldcode"], level["code"])


def add_levels_worldcode(worldcode: str) -> RequestAction:
    def action(request: Request) -> Request:
        levels = request.ellatu.db.level.get(worldcode=worldcode)
        for level in levels:
            request.levels[level_code_doc(level)] = level
        return request
    return action

def print_level_info() -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return trace(request, "No level selected")

        request.add_message(ParagraphMessage(
            f"**{request.level['title']} [{level_code_doc(request.level)}]**\n" +
            request.level['desc']))
        return request
    return action

def print_levels() -> RequestAction:
    def action(request: Request) -> Request:
        res = ""
        for levelcode, level in request.levels.items():
            res += f"{levelcode}: {level['title']}"
        return trace(request, res)
    return action

def print_worlds() -> RequestAction:
    def action(request: Request) -> Request:
        worlds = request.ellatu.db.world.get()
        res = ""
        for world in worlds:
            res += f"{world['code']}: {world['title']}\n"
        return trace(request, res)
    return action


# def move_users(levelcode: str) -> RequestAction:
#     def action(request: Request) -> Request:
#         for username, user in request.users.items():


def save_submit() -> RequestAction:
    def action(request: Request) -> Request:

        if request.level is None:
            return terminate_request(request, "Invalid level")

        for userid, codeblocks in request.codeblocks.items():
            if not request.ellatu.db.workplace.add_submission(
                userid,
                request.level["worldcode"],
                codeblocks
            ):
                terminate_request(request, "Unable to save submit")
        return request
    return action


def save_solution() -> RequestAction:
    def action(request: Request) -> Request:
        if not request.alive:
            return terminate_request(request, "The request is already dead")
        if request.level is None:
            return terminate_request(request, "Invalid level")
        for userid, _ in request.codeblocks.items():
            if not request.ellatu.db.solution.add_solution(
                userid,
                request.level['worldcode'],
                request.level['code'],
                3
            ):
                return terminate_request(request, "Unable to save solution")

        return trace(request, "The scores have been saved")
    return action



def sequence(actions: List[RequestAction]) -> RequestAction:

    def action(request: Request) -> Request:
        for action in actions:
            request = action(request)
            if not request.alive:
                break
        return request

    return action

class EllatuPipeline:

    def on_submit(self, request: Request) -> Request:
        return request

    def on_run(self, request: Request) -> Request:
        return request

class Ellatu:

    def __init__(self, ellatu_db: EllatuDB):
        self.on_submit_workflow = const_workflow
        self.on_run_workflow = const_workflow
        self.db = ellatu_db

    def user_move(self, userkey: UserKey, levelcode: str) -> Request:
        request = Request(self)
        return sequence([
            add_users([userkey]),
            localize_by_code(levelcode),
            move_users(),
            add_msg(TextMessage("**You have been moved to:**")),
            print_level_info()
        ])(request)

    def submit(self, userkey: UserKey, codeblocks: List[str]) -> Request:
        request = Request(self)
        return sequence([
            add_users([userkey]),
            assign_codeblocks({userkey: codeblocks}),
            localize_by_user(userkey),
            self.on_submit_workflow,
            save_submit(),
            add_msg(TextMessage("**The codeblocks was added to:**")),
            print_level_info()
        ])(request)

    def run(self, userkeys: List[UserKey]) -> Request:
        request = Request(self)
        return sequence([
            add_users(userkeys),
            localize_by_user(userkeys[0]),
            assign_from_workplace(userkeys),
            add_msg(TextMessage("**Running following blocks:**")),
            print_codeblocks(),
            self.on_run_workflow,
            save_solution()
        ])(request)

    def user_connected(self, userkey: UserKey, username: str) -> bool:
        return self.db.user.open_user(userkey, username) is not None

    def get_worlds(self) -> Request:
        request = Request(self)

        return sequence([
            print_worlds()
        ])(request)

    def get_levels(self, worldcode: str) -> Request:
        request = Request(self)

        return sequence([
            add_levels_worldcode(worldcode),
            print_levels()
        ])(request)

    def sign_for_user(self, userkey: UserKey) -> Request:
        request = Request(self)
        return sequence([
            localize_by_user(userkey),
            print_level_info()
        ])(request)
