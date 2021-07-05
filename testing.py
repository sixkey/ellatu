from ellatu_db import EllatuDB
from pprint import pprint
from ellatu import Codeblock, Ellatu

ellatudb = EllatuDB("localhost", 27017)
ellatu = Ellatu(ellatudb)

ellatu.user_connected("karl")
for cursor in ellatudb.user.collection.find():
    pprint(cursor)

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
