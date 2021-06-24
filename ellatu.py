from ellatu_db import EllatuDB
from typing import List, Callable, Optional

class Codeblock:

    def __init__(self, userid: str, code: str):
        self.userid = userid
        self.code = code

class Message:

    def __init__(self):
        pass

class TextMessage(Message):

    def __init__(self, message: str):
        super().__init__()
        self.message = message

class Submission:

    def __init__(self, codeblocks):
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

    def __init__(self):
        self.on_submit_workflow = kill_submission
        self.on_run_workflow = kill_submission

    def on_submit(self, userid: str, codeblock: str) -> Submission:
        block = Codeblock(userid, codeblock)
        submission = Submission([block])
        submission = self.on_submit_workflow(submission)
        return submission

    def on_run(self, users: str):
        pass



