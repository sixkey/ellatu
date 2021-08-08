from pprint import pprint
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar
from pymongo import MongoClient
from datetime import datetime

from pymongo.collection import ReturnDocument

DB_DEV = "ellatu_dev"

COL_USERS = "users"
COL_LEVELS = "levels"
COL_WORKPLACES = "workplaces"
COL_SOLUTIONS = "solutions"
COL_WORLDS = "worlds"

MAX_CODEBLOCKS = 3

Value = Any

T = TypeVar("T")

Collection = Any
Document = Dict[Any, Any]
MongoId = str

ClientService = str
ClientId = str
UserKey = Tuple[ClientService, ClientId]
Worldcode = str
Levelcode = str
LevelKey = Tuple[Worldcode, Levelcode]


###############################################################################
# Misc
###############################################################################

def int_in_range(value: int, min_value: Optional[int],
                 max_value: Optional[int]) -> bool:
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def get_userkey(doc: Document) -> UserKey:
    return (doc["client_ser"], doc['client_id'])


def get_levelkey(level: Document) -> LevelKey:
    return (level["worldcode"], level["code"])

###############################################################################
# Validation
###############################################################################


def val_error(trace: List[str], message: str) -> bool:
    trace.append(message)
    return False


class Validator(Generic[T]):

    def __init__(self, message: str = "invalid value"):
        self.message = message

    def pred(self, value: T, trace: List[str]) -> bool:
        return False

    def validate(self, value: T) -> bool:
        trace: List[str] = []
        if not self.pred(value, trace):
            print("Following value is invalid: ")
            pprint(value)
            print("\n".join(trace))
            return False
        return True


class StringValidator(Validator[str]):

    def __init__(self, min_size: Optional[int] = None,
                 max_size: Optional[int] = None,
                 charset: Optional[Set[str]] = None):
        super().__init__(message="invalid string")
        self.min_size = min_size
        self.max_size = max_size
        self.charset = charset

    def pred(self, value: str, trace: List[str]) -> bool:
        if not int_in_range(len(value), self.min_size, self.max_size):
            return val_error(trace, "Invalid string length")
        if self.charset:
            for c in value:
                if c not in self.charset:
                    return val_error(trace, "Character not in the allowed " +
                                     "charset")
        return True


class IntegerValidator(Validator[int]):
    def __init__(self, min_value: Optional[int] = None,
                 max_value: Optional[int] = None):
        super().__init__(message="invalid number")
        self.min_value = min_value
        self.max_value = max_value

    def pred(self, value: int, trace: List[str]) -> bool:
        if not isinstance(value, int):
            return val_error(trace, "Not integer")
        if not int_in_range(value, self.min_value, self.max_value):
            return val_error(trace, f"Not in the range {self.min_value}"
                             + f" {self.max_value}")
        return True


class RefKeyValidator(Validator[T]):

    def __init__(self, collection: Collection, atr_name: str):
        super().__init__("key not present in collection")
        self.collection = collection
        self.atr_name = atr_name

    def pred(self, value: T, trace: List[str]) -> bool:
        doc = {}
        doc[self.atr_name] = value
        if not self.collection.find_one(doc):
            return val_error(trace, f"Reference ({value}) not in the"
                             + " collection")
        return True


class RefValidator(Validator[Document]):

    def __init__(self, collection: Collection, keys: Dict[str, str]):
        super().__init__("value not present")
        self.collection = collection
        self.keys = keys

    def pred(self, value: Document, trace: List[str]) -> bool:
        mask = {}
        for key in self.keys:
            if key not in value:
                return val_error(trace, "The key referenced is not in the"
                                 + " value")
            mask[self.keys[key]] = value[key]
        if not self.collection.find_one(mask):
            return val_error(trace, "The reference is not present")
        return True


class PrimaryKeyValidator(Validator[Document]):

    def __init__(self, collection: Collection, keys: List[str]):
        super().__init__("primary key already present")
        self.collection = collection
        self.keys = keys

    def pred(self, value: Document, trace: List[str]) -> bool:
        mask = {}
        for key in self.keys:
            if key not in value:
                return val_error(trace, "The key that is part of the primary"
                                 + "key is not present")
            mask[key] = value[key]
        if self.collection.find_one(mask):
            return val_error(trace, "The primary key is already present")

        return True


class ReqValidator(Validator[Optional[Any]]):

    def __init__(self) -> None:
        super().__init__("value is required")

    def pred(self, value: Optional[Any], _: List[str]) -> bool:
        return value is not None


class ReqFieldsValidator(Validator[Document]):
    def __init__(self, keys: List[str]):
        super().__init__("required value is not present")
        self.keys = keys

    def pred(self, value: Document, trace: List[str]) -> bool:
        for key in self.keys:
            if key not in value or value[key] is None:
                return val_error(trace, f"The required value ({key}) is " +
                                 "not present")
        return True


class DictValidator(Validator[Document]):

    def __init__(self, scheme: Dict[str, Validator[Any]]):
        super().__init__("value not in scheme")
        self.scheme = scheme

    def pred(self, value: Document, trace: List[str]) -> bool:

        if not isinstance(value, dict):
            return val_error(trace, "The value is not dictionary")

        for key in self.scheme:
            if key not in value:
                return val_error(trace, f"Key {key} is not present")
            if not self.scheme[key].pred(value[key], trace):
                return val_error(trace, f"The validator failed on key: {key}")

        return True


class SequentialValidator(Validator[T]):

    def __init__(self, validators: List[Validator[T]]):
        super().__init__("one of validators failed")
        self.validators = validators

    def pred(self, value: T, trace: List[str]) -> bool:
        for validator in self.validators:
            if not validator.pred(value, trace):
                return val_error(trace, "Sequence failed")
        return True


class ListValidator(Validator[List[T]]):

    def __init__(self, validator: Optional[Validator[T]]):
        super().__init__("one of the values is not valid")
        self.validator = validator

    def pred(self, value: List[T], trace: List[str]) -> bool:
        if not isinstance(value, list):
            return val_error(trace, "The value is not list")
        if self.validator is not None:
            for subvalue in value:
                if not self.validator.pred(subvalue, trace):
                    return val_error(trace, "ListValidator failed on "
                                     + f"{subvalue}")
        return True


class OptionalValidator(Validator[T]):

    def __init__(self, validator: Validator[T]):
        super().__init__("the value is not none")
        self.validator = validator

    def pred(self, value: T, trace: List[str]) -> bool:
        if value is None:
            return True
        return self.validator.pred(value, trace)


###############################################################################
# Models
###############################################################################

Kwargs = Any


class Model:
    def __init__(self, collection: Collection):
        self.collection = collection
        self.validator: Optional[Validator[Document]] = None
        self.doc_validator: Optional[Validator[Document]] = None
        self.defaults: Optional[Document] = None

    def build_dict(self, **kwargs: Kwargs) -> Document:
        doc = {}
        for key, value in kwargs.items():
            doc[key] = value
        return doc

    def add(self, **kwargs: Kwargs) -> Optional[Document]:
        doc = self.build_dict(**kwargs)
        return self.d_add(doc)

    def _add_def(self, doc: Document) -> Document:
        if self.defaults is not None:
            for key, value in self.defaults.items():
                if key not in doc:
                    if callable(value):
                        doc[key] = value()
                    else:
                        doc[key] = value
        return doc

    def d_add(self, doc: Document) -> Optional[Document]:
        self._add_def(doc)
        if self.validator is not None and not self.validator.validate(doc):
            return None
        return self.collection.insert_one(doc)

    def d_update(self, find: Document, doc: Document,
                 upsert=True) -> Optional[Document]:

        value = self.collection.find_one_and_update(
            find, {"$set": doc}, upsert=False,
            return_document=ReturnDocument.AFTER)
        if value:
            return value

        if not upsert:
            return None

        return self.d_add({**find, **doc})

    def d_rewrite(self, keys: List[str], doc: Document) -> Optional[Document]:
        find = {}
        for key in keys:
            find[key] = doc[key]
        return self.d_update(find, doc, upsert=True)

    def get(self, **kwargs: Kwargs) -> List[Document]:
        query = self.build_dict(**kwargs)
        return self.collection.find(query)

    def get_one(self, **kwargs: Kwargs) -> Optional[Document]:
        query = self.build_dict(**kwargs)
        return self.collection.find_one(query)

    def get_by_id(self, id: MongoId) -> Optional[Document]:
        return self.collection.find_one({"_id": id})

    def exists(self, **kwargs: Kwargs) -> bool:
        return self.get_one(**kwargs) is not None


class User(Model):

    def __init__(self, collection: Collection):
        super().__init__(collection)
        self.validator = SequentialValidator([
            ReqFieldsValidator(["client_ser", "client_id", "username"]),
            PrimaryKeyValidator(collection, ["client_ser", "client_id"]),
            DictValidator({
                "client_ser": StringValidator(min_size=1, max_size=64),
                "client_id": StringValidator(min_size=1, max_size=64),
                "username": StringValidator(min_size=1, max_size=64),
                "levelcode": OptionalValidator(codeValidator),
                "worldcode": OptionalValidator(codeValidator)
            })
        ])
        self.defaults = {
            "levelcode": None,
            "worldcode": None
        }

    def get_user(self, userkey: UserKey) -> Optional[Document]:
        return self.get_one(client_ser=userkey[0], client_id=userkey[1])

    def get_users(self, userkeys: List[UserKey]) -> List[Document]:
        users = []
        for userkey in userkeys:
            user = self.get_user(userkey)
            if user is not None:
                users.append(user)
        return users

    def add_user(self, userkey: UserKey, username: str) -> Optional[Document]:
        return self.add(client_ser=userkey[0], client_id=userkey[1],
                        username=username)

    def open_user(self, userkey: UserKey, username: str) -> Optional[Document]:
        user = self.get_user(userkey)
        if user:
            return user
        return self.add_user(userkey, username)

    def move_user(self, userkey: UserKey, worldcode: str,
                  levelcode: str) -> Optional[Document]:
        return self.collection.update_one(
            {'client_ser': userkey[0],
             'client_id': userkey[1]},
            {"$set": {"levelcode": levelcode, "worldcode": worldcode}},
            upsert=False
        )


codeValidator = StringValidator(min_size=1, max_size=10)
titleValidator = StringValidator(min_size=1, max_size=64)


class World(Model):

    def __init__(self, collection: Collection):
        super().__init__(collection)
        self.doc_validator = SequentialValidator([
            ReqFieldsValidator(["title", "code", "tags", "prereqs"]),
            DictValidator({
                "title": titleValidator,
                "code": codeValidator,
                "tags": ListValidator(StringValidator(min_size=4,
                                                      max_size=32)),
                "prereqs": ListValidator(RefKeyValidator(collection, "code")),
            })
        ])
        self.validator = SequentialValidator([
            self.doc_validator,
            PrimaryKeyValidator(collection, ["code"])
        ])
        self.defaults = {
            "tags": [],
            "prereqs": [],
        }


class Level(Model):

    def __init__(self, collection: Collection, worlds: Collection):
        super().__init__(collection)
        self.doc_validator = SequentialValidator([
            ReqFieldsValidator(
                ["title", "code", "worldcode", "prereqs", "pipeline"]),
            DictValidator({
                "title": titleValidator,
                "desc": StringValidator(),

                "code": codeValidator,
                "worldcode": RefKeyValidator(worlds, "code"),

                "prereqs": ListValidator(RefKeyValidator(collection, "code")),

                "pipeline": StringValidator(min_size=4, max_size=16),
                "attrs": DictValidator({}),
                "tests": ListValidator(None),

                "tags": ListValidator(StringValidator(min_size=4,
                                                      max_size=32)),
            })
        ])
        self.validator = SequentialValidator([
            PrimaryKeyValidator(collection, ["worldcode", "code"]),
            self.doc_validator
        ])
        self.defaults = {
            "desc": "",
            "tags": [],
            "prereqs": [],
            "tests": [],
            "attrs": {}
        }

    def get_by_code(self, worldcode: str,
                    levelcode: str) -> Optional[Document]:
        return self.get_one(worldcode=worldcode, code=levelcode)


CodeblockValidator = StringValidator(min_size=1, max_size=4096)


class Workplace(Model):

    def __init__(self, collection: Collection, users: Collection,
                 worlds: Collection):
        super().__init__(collection)
        self.doc_validator = SequentialValidator([
            ReqFieldsValidator(["user", "worldcode", "submissions"]),
            DictValidator({
                "user": RefKeyValidator(users, "_id"),
                "worldcode": RefKeyValidator(worlds, "code"),
                "submissions": ListValidator(
                    ListValidator(CodeblockValidator)),
                "bench": ListValidator(CodeblockValidator)
            })
        ])
        self.validator = SequentialValidator([
            PrimaryKeyValidator(collection, ["user", "worldcode"]),
            self.doc_validator
        ])
        self.defaults = {"submissions": [], "bench": []}

    def add_submission(self, userid: MongoId, worldcode: str,
                       submission: List[str]) -> Optional[Document]:
        doc = self.build_dict(user=userid, worldcode=worldcode)
        if self.collection.find_one(doc) is None:
            if self.d_add(doc) is None:
                return None

        document = self.collection.find_one(doc)

        if document is None:
            return None

        submissions = document["submissions"]
        submissions.append(submission)
        if len(submissions) > MAX_CODEBLOCKS:
            submissions = submissions[len(submissions) - MAX_CODEBLOCKS:]
        return self.collection \
            .update_one(doc, {'$set': {"submissions": submissions}})

    def get_workplaces(self, userids: List[MongoId],
                       worldcode: str) -> List[Document]:
        return self.collection.find({"worldcode": worldcode,
                                     "user": {"$in": userids}})

    def get_codeblocks(self, userids: List[MongoId],
                       worldcode: str) -> Dict[MongoId, List[str]]:
        workplaces = self.get_workplaces(userids, worldcode)
        result = {}
        for workplace in workplaces:
            submissions = workplace["submissions"]
            codeblocks = []
            if submissions is not None and len(submissions) != 0:
                codeblocks = submissions[-1]
            result[workplace["user"]] = codeblocks
        return result

    def get_workbenches(self, userids: List[MongoId],
                        worldcode: str) -> Dict[MongoId, List[str]]:
        result = {}
        for workplace in self.get_workplaces(userids, worldcode):
            result[workplace["user"]] = workplace["bench"]
        return result

    def set_workbenches(self, codeblocks: Dict[MongoId, List[str]],
                        worldcode: str):
        results = []
        for userid, blocks in codeblocks.items():
            res = self.d_update(
                {"user": userid, "worldcode": worldcode}, {"bench": blocks})
            print(res)
            if res is not None:
                results.append(res)
        return results


class Solution(Model):

    def __init__(self, collection: Collection, users: Collection,
                 levels: Collection, worlds: Collection):
        super().__init__(collection)
        self.doc_validator = SequentialValidator([
            ReqFieldsValidator(
                ["user", "levelcode", "worldcode", "mark", "date"]),
            DictValidator({
                "user": RefKeyValidator(users, "_id"),
                "levelcode": RefKeyValidator(levels, "code"),
                "worldcode": RefKeyValidator(worlds, "code"),
                "mark": IntegerValidator(0, 3),
                # TODO: date validator
            }),
            RefValidator(levels, {"levelcode": "code",
                                  "worldcode": "worldcode"})
        ])
        self.validator = SequentialValidator([
            PrimaryKeyValidator(
                collection, ['user', 'levelcode', 'worldcode']),
            self.doc_validator
        ])
        self.defaults = {
            "date": datetime.now
        }

    def add_solution(self, userid: MongoId, worldcode: str, levelcode: str,
                     mark: int) -> Optional[Document]:
        sol_doc = self.build_dict(user=userid, levelcode=levelcode,
                                  worldcode=worldcode, mark=mark)
        return self.d_rewrite(['user', 'levelcode', 'worldcode'], sol_doc)

    def get_solutions(self, userid: MongoId, worldcode: Optional[str] = None) \
            -> List[Document]:
        query = {"user": userid}
        if worldcode is not None:
            query['worldcode'] = worldcode
        return self.collection.find(query)


class EllatuDB:

    def __init__(self, host: str, port: Optional[int] = None,
                 db_name: str = DB_DEV):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.user = User(self.db[COL_USERS])
        self.world = World(self.db[COL_WORLDS])
        self.level = Level(self.db[COL_LEVELS], self.db[COL_WORLDS])
        self.workplace = Workplace(self.db[COL_WORKPLACES],
                                   self.db[COL_USERS],
                                   self.db[COL_WORLDS])
        self.solution = Solution(self.db[COL_SOLUTIONS],
                                 self.db[COL_USERS],
                                 self.db[COL_LEVELS],
                                 self.db[COL_WORLDS])
