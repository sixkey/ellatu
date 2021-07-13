import re
from ellatu_db import EllatuDB
from pprint import pprint
from ellatu import Ellatu, Request, trace

print(re.match(r"^\s*!(\w+)[\s\S]*", "!ahoj").group(1))

if False:
    ellatudb = EllatuDB("localhost", 27017)
    ellatu = Ellatu(ellatudb)

    ellatudb.world.add(title="tuturial", code="tutori")
    world = ellatudb.world.get_one(code="tutori")
    ellatudb.level.add(title="tutorial", worldcode="tutori", code="tutori")

    ellatu.user_connected("karl")
    for cursor in ellatudb.user.collection.find():
        pprint(cursor)

    ellatu.user_move("karl", "tutori")

    def on_submit(request: Request) -> Request:

        codeblocks = []

        trace(request, "Compilation")

        for _ , users_blocks in request.codeblocks.items():
            codeblocks += users_blocks

        pprint(codeblocks)
        return request


    ellatu.on_submit_workflow = on_submit

    print(str(ellatu.submit("karl", ["hello world from karl"])))

    for cursor in ellatudb.workplace.collection.find():
        pprint(cursor)

    print(str(ellatu.run(["karl"])))


# pprint(ellatudb.user.get_user("martin"))
# pprint(ellatudb.user.add_user("martin"))
# pprint(ellatudb.user.add_user("mar"))
#
# ellatudb.level.add(title="Hell", code="000h3ll000")
#
# print("Getting user")
# user = ellatudb.user.get_user("martin")
# pprint(user)
# print("Getting level")
# level = ellatudb.level.get_one(code="000h3ll000")
# pprint(level)
# print("Setting up world")
# ellatudb.workplace.add_submission(user["_id"], level["_id"], "hello world")
#
# for cursor in ellatudb.workplace.collection.find():
#     pprint(cursor)
#
# ellatudb
