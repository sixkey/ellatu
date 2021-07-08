from ellatu_db import Document, EllatuDB, MongoId
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

    def __init__(self, message: str, message_type: MessageType = MessageType.LOG):
        super().__init__(message_type)
        self.message = message

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

def add_users(usernames: List[str]) -> RequestAction:
    def action(request: Request) -> Request:
        names_set = set(usernames)
        users = request.ellatu.db.user.get_users(usernames)
        for user in users:
            names_set.remove(user["username"])
            request.users[user["username"]] = user
        if names_set:
            terminate_request(request,
                              f"The users were not found {str(names_set)}")
        return request
    return action


def assign_codeblocks(assignments: Dict[str, List[str]]) -> RequestAction:
    def action(request: Request) -> Request:
        for username, codeblocks in assignments.items():
            if username not in request.users:
                request.messages.append(TextMessage(
                    f"{username} is not in the request"))
                continue
            user = request.users[username]
            request.codeblocks[user["_id"]] = codeblocks
        return request
    return action


def assign_from_workplace(usernames: List[str]) -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request,
                                     "The request doesn't have valid \
                                        level set")
        users = request.ellatu.db.user.get_users(usernames)
        userids = [u["_id"] for u in users]
        codeblocks = request.ellatu.db.workplace\
            .get_codeblocks(userids, request.level["worldcode"])

        for userid, blocks in codeblocks.items():
            request.codeblocks[userid] = blocks
        return request
    return action


def localize_by_user(username: str) -> RequestAction:
    def action(request: Request) -> Request:
        user = request.ellatu.db.user.get_user(username)
        if user is None:
            return terminate_request(request, "The user was not found")
        request.level = request.ellatu.db.level.get_by_code(user["levelcode"])
        return request
    return action


def localize_by_code(levelcode: str) -> RequestAction:
    def action(request: Request) -> Request:
        level = request.ellatu.db.level.get_by_code(levelcode)
        if not level:
            return terminate_request(request, "Invalid level code")
        request.level = level
        return request
    return action


def move_users() -> RequestAction:
    def action(request: Request) -> Request:
        if not request.level:
            return terminate_request(request, "Invalid level")
        for username, _ in request.users.items():
            request.ellatu.db.user.move_user(username, request.level["code"])
        return request
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


def sequence(actions: List[RequestAction]) -> RequestAction:

    def action(request: Request) -> Request:
        for action in actions:
            request = action(request)
            if not request.alive:
                break
        return request

    return action


class Ellatu:

    def __init__(self, ellatu_db: EllatuDB):
        self.on_submit_workflow = const_workflow
        self.on_run_workflow = const_workflow
        self.db = ellatu_db

    def user_move(self, username: str, levelcode: str) -> Request:
        request = Request(self)
        return sequence([
            add_users([username]),
            localize_by_code(levelcode),
            move_users()
        ])(request)

    def submit(self, username: str, codeblocks: List[str]) -> Request:
        request = Request(self)
        return sequence([
            add_users([username]),
            assign_codeblocks({username: codeblocks}),
            localize_by_user(username),
            self.on_submit_workflow,
            save_submit()
        ])(request)

    def run(self, usernames: List[str]) -> Request:
        request = Request(self)
        return sequence([
            add_users(usernames),
            localize_by_user(usernames[0]),
            assign_from_workplace(usernames),
            self.on_run_workflow
        ])(request)

    def user_connected(self, username: str) -> bool:
        return self.db.user.open_user(username) is not None

    def get_worlds(self) -> Request:


    def get_levels(self, worldcode: str) -> Request:
        request = Request(self)

        return sequence([
            add_levels_worldcode(worldcode),

        ])(request)

        return request
