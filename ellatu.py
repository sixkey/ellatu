from ellatu_db import Document, EllatuDB, MongoId
from typing import Dict, List, Callable, Optional, Any

class Codeblock:

    def __init__(self, user: Document, code: List[str]):
        self.user = user
        self.code = code

class Message:

    def __init__(self):
        pass

class TextMessage(Message):

    def __init__(self, message: str):
        super().__init__()
        self.message = message

class Submission:

    def __init__(self, codeblocks: Dict[MongoId, Codeblock], context: Any = None):
        self.context = context
        self.codeblocks = codeblocks
        self.messages: List[Message] = []
        self.alive = True
        self.box = None

SubmissionAction = Callable[[Submission], Submission]

def const_workflow(submission: Submission) -> Submission:
    return submission

def kill_submission(submission: Submission) -> Submission:
    submission.alive = False
    return submission


def sequence(actions: List[SubmissionAction]) -> SubmissionAction:

    def action(submission: Submission) -> Submission:
        for action in actions:
            submission = action(submission)
            if not submission.alive:
                break
        return submission

    return action


class Ellatu:

    def __init__(self, ellatu_db: EllatuDB):
        self.on_submit_workflow = kill_submission
        self.on_run_workflow = kill_submission
        self.db = ellatu_db

    def submit(self, username: str, codeblocks: List[str], worldid: str) -> Submission:
        user = self.db.user.get_user(username)
        submission = Submission({user["_id"]:Codeblock(user, codeblocks)}, self)
        submission = self.on_submit_workflow(submission)
        return submission

    def run(self, usernames: List[str], worldid: str):
        users = self.db.user.get_users(usernames)
        if len(users) != len(usernames):
            return None

        userids = [u["_id"] for u in users]
        codeblocks = self.db.workplace.get_codeblocks(userids, worldid)

        blocks = {}
        for user in users:
            blocks[user["_id"]] = Codeblock(user, codeblocks[user["_id"]])
        submission = Submission(blocks, self)
        submission = self.on_run_workflow(submission)
        return submission

    def user_connected(self, username: str):
        self.db.user.open_user(username)





