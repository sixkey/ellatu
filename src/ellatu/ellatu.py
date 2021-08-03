from collections import deque
import os
import re
from typing import Deque, Dict, List, Callable, Optional, Any, Set, Tuple, TypeVar
from enum import Enum
from datetime import datetime, timedelta
from random import randint
import pygraphviz

from . import image_editing as imge
from .ellatu_db import Document, EllatuDB, MongoId, UserKey, LevelKey, get_levelkey, get_userkey

###############################################################################
# Types
###############################################################################

F = TypeVar('F', bound=Callable[..., Any])

###############################################################################
# Misc.
###############################################################################


def level_code_parts(worldcode: str, levelcode: str) -> str:
    return f"{worldcode}-{levelcode}"


def level_code_doc(level: Document) -> str:
    return level_code_parts(level["worldcode"], level["code"])


def level_code_key(levelkey: LevelKey) -> str:
    return level_code_parts(levelkey[0], levelkey[1])

###############################################################################
# Messages
###############################################################################


class MessageType(Enum):
    LOG = 1
    TRACE = 2
    ERROR = 3


class Message:

    def __init__(self, message_type: MessageType):
        self.message_type = message_type

    def prefix(self, message_body: str) -> str:
        return f"[{self.message_type}]: {message_body}"


class MessageSegment(Message):

    def __init__(self, title: str, message_type: MessageType = MessageType.LOG):
        super().__init__(message_type)
        self.title = title

    def __str__(self) -> str:
        return self.prefix(self.title)


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


class ImageMessage(Message):
    def __init__(self, alt_text: str, location: str,
                 message_type: MessageType = MessageType.LOG) -> None:
        super().__init__(message_type)
        self.alt_text = alt_text
        self.location = location

    def __str__(self) -> str:
        return self.prefix(f"Image [{self.location}][{self.alt_text}]")


###############################################################################
# Request
###############################################################################


class Request:

    def __init__(self, ellatu: "Ellatu",
                 codeblocks: Optional[Dict[MongoId, List[str]]] = None):

        self.id = ellatu.get_req_id()

        self.ellatu = ellatu
        self.alive = True

        self.messages: List[Message] = []

        self.level: Optional[Document] = None
        self.levels: Dict[LevelKey, Document] = {}

        self.users: Dict[UserKey, Document] = {}

        self.codeblocks: Dict[MongoId, List[str]
                              ] = codeblocks if codeblocks else {}

        self.data: Dict[str, Any] = {}

        self._on_resolved_actions: List['RequestAction'] = []

    def add_on_res(self, action: 'RequestAction') -> None:
        self._on_resolved_actions.append(action)

    def on_resolved(self) -> 'Request':
        return pipeline_sequence(self._on_resolved_actions)(self)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def __str__(self) -> str:
        res = f"Request: [{'Alive' if self.alive else 'Dead'}]\n"
        for message in self.messages:
            res += str(message) + '\n'
        return res


RequestAction = Callable[[Request], Request]


# Message Utils ###############################################################


def log_msg(request: Request, message: str,
            message_type: MessageType = MessageType.LOG) -> Request:
    request.add_message(TextMessage(message, message_type=message_type))
    return request


def trace(request: Request, message: str) -> Request:
    return log_msg(request, message, MessageType.TRACE)


def log(request: Request, message: str) -> Request:
    return log_msg(request, message, MessageType.LOG)


def error(request: Request, message: str) -> Request:
    return log_msg(request, message, MessageType.ERROR)


def add_msg(message: Message) -> RequestAction:
    def action(request: Request) -> Request:
        request.add_message(message)
        return request
    return action


def terminate_request(request: Request, message: str) -> Request:
    request.alive = False
    return error(request, message)


def const_action(request: Request) -> Request:
    return request


def kill_request(request: Request) -> Request:
    return terminate_request(request, "The request has been killed")


# Action utils ################################################################

ExtRequestAction = Callable[..., Request]


def data_action(keys: List[str]) \
        -> Callable[[ExtRequestAction], RequestAction]:
    def wrapper(func: ExtRequestAction) -> RequestAction:
        def action(request: Request) -> Request:
            values = []
            for key in keys:
                if key not in request.data:
                    return terminate_request(request, "Internal error")
                values.append(request.data[key])
            return func(request, *values)
        return action
    return wrapper


def pipeline_sequence(actions: List[RequestAction]) -> RequestAction:

    def action(request: Request) -> Request:
        for action in actions:
            request = action(request)
            if not request.alive:
                break
        return request

    return action


def pipeline_tree(tree: Dict[str, RequestAction]) -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request, "No level set")
        if request.level['pipeline'] not in tree:
            return terminate_request(request, "Unknown pipeline")
        return tree[request.level['pipeline']](request)
    return action

# File in action ##############################################################


def remove_files(filenames: List[str]) -> RequestAction:
    def action(request: Request) -> Request:
        for filename in filenames:
            os.remove(filename)
        return request
    return action

# User actions ################################################################


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


def move_users() -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request, "Invalid level")
        for userkey, _ in request.users.items():
            if not request.ellatu.db.user.move_user(userkey, request.level["worldcode"], request.level["code"]):
                return terminate_request(request, "Unable to move user")
        return request
    return action

# Codeblocks actions ##########################################################


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


def limit_users(number: Optional[int]) -> RequestAction:
    def action(request: Request) -> Request:
        if number is None:
            return request
        if len(request.codeblocks.keys()) > number:
            return terminate_request(
                request,
                f"There were more than {number} users")
        return request
    return action


def limit_codeblocks(number: Optional[int]) -> RequestAction:
    def action(request: Request) -> Request:
        if number is None:
            return request
        for _, codeblocks in request.codeblocks.items():
            if len(codeblocks) > number:
                return terminate_request(
                    request,
                    f"There were more then {number} codeblocks"
                )
        return request
    return action


def limit_lines(number: Optional[int]) -> RequestAction:
    def action(request: Request) -> Request:
        if number is None:
            return request
        for _, codeblocks in request.codeblocks.items():
            for codeblock in codeblocks:
                if len(list(filter(lambda x: x.strip() != '',
                                   codeblock.splitlines()))) > number:
                    return terminate_request(
                        request,
                        f"A codeblock was longer than {number} lines"
                    )
        return request
    return action


def limit_columns(number: Optional[int]) -> RequestAction:
    def action(request: Request) -> Request:
        if number is None:
            return request
        for _, codeblocks in request.codeblocks.items():
            for codeblock in codeblocks:
                for line in codeblock.splitlines():
                    if len(line.rstrip()) > number:
                        return terminate_request(
                            request,
                            f"Line was longer than {number} characters"
                        )
        return request
    return action


# Localize actions ############################################################


def parse_level_code(code: str) -> Tuple[Optional[str], str]:
    match = re.match(r'^(\w+)\-(\w+)$', code)
    if match:
        return match.group(1), match.group(2)
    return None, code


def localize_by_user(userkey: UserKey) -> RequestAction:
    def action(request: Request) -> Request:
        user = request.ellatu.db.user.get_user(userkey)
        if user is None:
            return terminate_request(request, "The user was not found")
        worldcode = user['worldcode']
        levelcode = user['levelcode']
        if worldcode is None or levelcode is None:
            return terminate_request(request, "The user is in no level")
        request.level = request.ellatu.db.level.get_by_code(worldcode,
                                                            levelcode)
        return request
    return action


def localize_by_code(worldcode: Optional[str], levelcode: Optional[str], userkey: Optional[UserKey] = None) -> RequestAction:
    def action(request: Request) -> Request:
        w_code = worldcode
        l_code = levelcode
        if w_code is None or l_code is None:
            if userkey is None:
                return terminate_request(request, "Part of code missing but no user")
            user = request.ellatu.db.user.get_user(userkey)
            if user is None:
                return terminate_request(request, "Part of code missing but user not found")
            if user['worldcode'] is None or user['levelcode'] is None:
                return terminate_request(request, "The user is in no level")
            if w_code is None:
                w_code = user['worldcode']
            if l_code is None:
                l_code = user['levelcode']
        level = request.ellatu.db.level.get_one(
            worldcode=w_code, code=l_code)
        if not level:
            return terminate_request(request, "Invalid level code")
        request.level = level
        return request
    return action


# Level/world actions #########################################################


def add_levels_worldcode(worldcode: str) -> RequestAction:
    def action(request: Request) -> Request:
        levels = request.ellatu.db.level.get(worldcode=worldcode)
        for level in levels:
            request.levels[get_levelkey(level)] = level
        return request
    return action


def add_local_levels() -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request, "Level not set")
        return add_levels_worldcode(request.level['worldcode'])(request)
    return action


def print_levels() -> RequestAction:
    @data_action(["beaten"])
    def action(request: Request, beaten: Set[Tuple[str, str]]) -> Request:
        res = ""
        for _, level in request.levels.items():
            title_str = f"**{level['title']}** [*{level_code_doc(level)}*]"

            locked = is_locked(beaten, level)
            done = is_beaten(beaten, level)
            title_wrapper = ('', '')
            if done:
                title_wrapper = ('__', '__')
            elif locked:
                title_wrapper = ('|| Locked: ', '||')
            res += title_wrapper[0] + title_str + title_wrapper[1]
            if level['prereqs']:
                res += f" needs {','.join(['_' + s + '_' for s in level['prereqs']])}"
            res += '\n'
        return trace(request, res)
    return action


def draw_levels(filename: str, userkey: Optional[UserKey] = None,
                include_worldcode: bool = True) -> RequestAction:
    @data_action(["beaten"])
    def action(request: Request, beaten: Set[Tuple[str, str]]) -> Request:
        dot = pygraphviz.AGraph(strict=False, directed=True)
        dot.node_attr["shape"] = "rect"
        dot.node_attr["style"] = "filled"
        target: Optional[Tuple[str, str]] = None

        if userkey is not None:
            if userkey not in request.users:
                return terminate_request(request, "User not loaded")
            user = request.users[userkey]
            target = (user['worldcode'], user['levelcode'])

        for _, level in request.levels.items():
            fillcolor = "white"
            if is_beaten(beaten, level):
                fillcolor = "#56de3e"
            if is_locked(beaten, level):
                fillcolor = "#d63c31"
            if target and target == (level['worldcode'], level['code']):
                fillcolor = "#5c70d6"
            code = level_code_doc(level)
            label = code if include_worldcode else level['code']
            dot.add_node(code,  fillcolor=fillcolor, label=label)
            for prereq in level['prereqs']:
                dot.add_edge(level_code_parts(
                    level['worldcode'], prereq), code)
        dot.layout(prog='dot')
        dot.draw(filename)
        imge.edit_image(filename, imge.edit_sequence([
            imge.expand_to_aspect(1),
            imge.expand(total=20)
        ]))
        return request
    return action


def print_level_info(
        header: Callable[[Document], Optional[str]] = lambda _: None,
        desc: bool = True) -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return trace(request, "No level selected")
        text = f"**{request.level['title']}** " + \
            f"[_{level_code_doc(request.level)}_]"

        header_text = header(request.level)
        if header_text:
            text += '\n' + header_text
        if desc:
            text += '\n' + request.level['desc']

        request.add_message(ParagraphMessage(text))
        return request
    return action


def format_world_title(world: Document) -> str:
    return f"**{world['title']}** [_{world['code']}_]\n"


def print_worlds() -> RequestAction:
    def action(request: Request) -> Request:
        worlds = request.ellatu.db.world.get()
        res = ""
        for world in worlds:
            res += format_world_title(world)
        return trace(request, res)
    return action


def print_world_info(worldcode: str) -> RequestAction:
    def action(request: Request) -> Request:
        world = request.ellatu.db.world.get_one(code=worldcode)
        if world is None:
            return terminate_request(request, "World not found")
        return trace(request, format_world_title(world))
    return action


def print_world_info_local() -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return terminate_request(request, "Level not set")
        return print_world_info(request.level['worldcode'])(request)
    return action


def print_world_info_user(userkey: UserKey) -> RequestAction:
    def action(request: Request) -> Request:
        if userkey not in request.users:
            return terminate_request(request, "User not in request")
        user = request.users[userkey]
        if not user['worldcode']:
            return terminate_request(request, "User not in world")
        return print_world_info(user['worldcode'])(request)
    return action

# Permission actions ##########################################################


def beaten_from_solutions(solutions: List[Document]) -> Set[Tuple[str, str]]:
    return set([(s['worldcode'], s['levelcode']) for s in solutions])


def is_locked(beaten: Optional[Set[Tuple[str, str]]], level: Document) -> bool:
    if beaten is None:
        return False
    for pre in level['prereqs']:
        if (level['worldcode'], pre) not in beaten:
            return True
    return False


def is_beaten(beaten: Optional[Set[Tuple[str, str]]], level: Document) -> bool:
    if beaten is None:
        return False
    return (level['worldcode'], level['code']) in beaten


def permission_check() -> RequestAction:
    def action(request: Request) -> Request:
        if request.level is None:
            return trace(request, "No level to check permission for")

        not_allowed_users = []
        for _, user in request.users.items():
            sols = request.ellatu.db.solution.get_solutions(
                user["_id"], request.level['worldcode'])
            beaten = beaten_from_solutions(sols)
            if is_locked(beaten, request.level):
                not_allowed_users.append(user['username'])
        if not_allowed_users:
            return terminate_request(
                request,
                "Users are not allowed to be in the level: " +
                f"{','.join(not_allowed_users)}")
        return request
    return action


# Submission actions ##########################################################


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


def load_beaten_by_user(userkey: UserKey) -> RequestAction:
    def action(request: Request) -> Request:
        user = request.ellatu.db.user.get_user(userkey)
        if user is None:
            return terminate_request(request, "User not found")
        solutions = request.ellatu.db.solution.get_solutions(user["_id"])
        beaten = set([(s['worldcode'], s['levelcode']) for s in solutions])
        request.data["beaten"] = beaten
        return request
    return action


###############################################################################
# Ellatu interface
###############################################################################

class TempFileStorage:

    def __init__(self, temp_folder: str = 'ellatu_temp'):
        self.temp_folder = temp_folder
        self.temp_files: Deque[Tuple[datetime, str]] = deque()

    def add_temp_filename(self, name: str) -> str:
        filename = os.path.join(self.temp_folder, name)
        self.temp_files.append((datetime.now(), filename))
        return filename

    def remove_temp_files(self) -> None:
        if not self.temp_files:
            return
        dumpdate = datetime.now() - timedelta(hours=1)
        while self.temp_files[0][0] < dumpdate:
            _, temp_file = self.temp_files.popleft()
            os.remove(temp_file)


class Ellatu:

    def __init__(self, ellatu_db: EllatuDB, temp_folder: str = 'ellatu_temp'):
        self.on_submit_workflow: RequestAction = const_action
        self.on_run_workflow: RequestAction = const_action
        self.header: Callable[[Document], Optional[str]] = lambda x: None
        self.db = ellatu_db
        self.temp_files = TempFileStorage(temp_folder=temp_folder)

    def user_move(self, userkey: UserKey, code: str) -> Request:
        request = Request(self)
        worldcode, levelcode = parse_level_code(code)
        return pipeline_sequence([
            add_users([userkey]),
            localize_by_code(worldcode, levelcode, userkey),
            permission_check(),
            move_users(),
            add_msg(TextMessage("**You have been moved to:**")),
            print_level_info(header=self.header)
        ])(request)

    def submit(self, userkey: UserKey, codeblocks: List[str]) -> Request:
        request = Request(self)
        return pipeline_sequence([
            add_users([userkey]),
            localize_by_user(userkey),
            permission_check(),
            assign_codeblocks({userkey: codeblocks}),
            self.on_submit_workflow,
            save_submit(),
            add_msg(MessageSegment("The codeblocks was added to:")),
            print_level_info(header=self.header, desc=False)
        ])(request)

    def run(self, userkeys: List[UserKey]) -> Request:
        request = Request(self)
        return pipeline_sequence([
            add_users(userkeys),
            localize_by_user(userkeys[0]),
            permission_check(),
            assign_from_workplace(userkeys),
            add_msg(MessageSegment("Running following blocks:")),
            print_codeblocks(),
            self.on_run_workflow,
            save_solution()
        ])(request)

    def user_connected(self, userkey: UserKey, username: str) -> bool:
        return self.db.user.open_user(userkey, username) is not None

    def get_worlds(self) -> Request:
        request = Request(self)
        return pipeline_sequence([
            add_msg(MessageSegment('Available worlds')),
            print_worlds()
        ])(request)

    def get_levels(self, userkey: UserKey,
                   worldcode: Optional[str] = None) -> Request:
        request = Request(self)
        return pipeline_sequence([
            pipeline_sequence(
                [localize_by_user(userkey), add_local_levels()]
            ) if worldcode is None else add_levels_worldcode(worldcode),
            load_beaten_by_user(userkey),
            add_msg(MessageSegment('Available levels in _mapper_')),
            print_levels()
        ])(request)

    def draw_map(self, userkey: UserKey, worldcode: Optional[str] = None) -> Request:

        # TODO: temp solution

        request = Request(self)

        filename = self.temp_files.add_temp_filename(
            str(request.id) + '-map.png'
        )
        request = pipeline_sequence([
            add_users([userkey]),
            pipeline_sequence(
                [localize_by_user(userkey), add_local_levels()]
            ) if worldcode is None else add_levels_worldcode(worldcode),
            load_beaten_by_user(userkey),
            draw_levels(filename, userkey=userkey, include_worldcode=False),
            print_world_info_user(
                userkey) if worldcode is None else print_world_info(worldcode),
            add_msg(ImageMessage('map', filename))
        ])(request)
        request.add_on_res(remove_files([filename]))
        return request

    def sign_for_user(self, userkey: UserKey) -> Request:
        request = Request(self)
        return pipeline_sequence([
            add_users([userkey]),
            localize_by_user(userkey),
            permission_check(),
            print_level_info(header=self.header)
        ])(request)

    def get_req_id(self) -> int:
        return randint(0, 1000000000000)


class EllatuPipeline:

    def on_submit(self) -> RequestAction:
        return const_action

    def on_run(self) -> RequestAction:
        return const_action
