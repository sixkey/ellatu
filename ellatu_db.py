from pprint import pprint
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar
from pymongo import MongoClient
from datetime import datetime

DB_DEV = "db_dev"

COL_USERS = "users"
COL_LEVELS = "levels"
COL_WORKPLACES = "workplaces"
COL_SOLUTIONS = "solutions"

MAX_CODEBLOCKS = 3

LEVELS = "levels"


Value = Any

T = TypeVar("T")

Collection = Any
Document = Optional[Any]
JSON = Dict[str, Any]

def int_in_range(value: int, min_value: Optional[int],
                 max_value: Optional[int]) -> bool:
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


###############################################################################
## Validation
###############################################################################

class Validator(Generic[T]):

    def __init__(self, message: str = "invalid value"):
        self.message = message

    def pred(self, value: T) -> bool:
        return False

    def validate(self, value: T) -> bool:
        if not self.pred(value):
            print(self.message)
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

    def pred(self, value: str) -> bool:
        if not int_in_range(len(value), self.min_size, self.max_size):
            return False
        if self.charset:
            for c in value:
                if c not in self.charset:
                    return False
        return True

class IntegerValidator(Validator[int]):
    def __init__(self, min_value: Optional[int] = None,
                max_value: Optional[int] = None):
        super().__init__(message="invalid number")
        self.min_value = min_value
        self.max_value = max_value

    def pred(self, value: int) -> bool:
        return isinstance(value, int) and \
            not int_in_range(value, self.min_value, self.max_value)


class RefKeyValidator(Validator[T]):

    def __init__(self, collection: Collection, atr_name: str):
        super().__init__("key not present in collection")
        self.collection = collection
        self.atr_name = atr_name

    def pred(self, value: T):
        doc = {}
        doc[self.atr_name] = value
        if not self.collection.find_one(doc):
            return False
        return True


class RefValidator(Validator[JSON]):

    def __init__(self, collection: Collection, keys: Dict[str, str]):
        super().__init__("value not present")
        self.collection = collection
        self.keys = keys

    def pred(self, value: JSON):
        mask = {}
        for key in self.keys:
            if key not in value:
                return False
            mask[self.keys[key]] = value[key]
        if not self.collection.find_one(mask):
            return False
        return True

class PrimaryKeyValidator(Validator[JSON]):

    def __init__(self, collection: Collection, keys: List[str]):
        super().__init__("primary key already present")
        self.collection = collection
        self.keys = keys

    def pred(self, value: JSON):
        mask = {}
        for key in self.keys:
            if key not in value:
                return False
            mask[key] = value[key]

        if self.collection.find_one(mask):
            return False

        return True

class ReqValidator(Validator[Optional[Any]]):

    def __init__(self):
        super().__init__("value is required")

    def pred(self, value: Optional[Any]) -> bool:
        return value is not None

class ReqFieldsValidator(Validator[JSON]):
    def __init__(self, keys: List[str]):
        super().__init__("required value is not present")
        self.keys = keys

    def pred(self, value: JSON):
        for key in self.keys:
            if key not in value or value[key] is None:
                return False
        return True

class DictValidator(Validator[JSON]):

    def __init__(self, scheme: Dict[str, Validator[Any]]):
        super().__init__("value not in scheme")
        self.scheme = scheme

    def pred(self, value: JSON) -> bool:
        for key in self.scheme:
            if key not in value:
                return False
            if not self.scheme[key].validate(value[key]):
                return False

        return True

class SequentialValidator(Validator[T]):

    def __init__(self, validators: List[Validator[T]]):
        super().__init__("one of validators failed")
        self.validators = validators

    def pred(self, value: T) -> bool:
        for validator in self.validators:
            if not validator.validate(value):
                return False
        return True

class ListValidator(Validator[List[T]]):

    def __init__(self, validator: Optional[Validator[T]]):
        super().__init__("one of the values is not valid")
        self.validator = validator

    def pred(self, value: List[T]) -> bool:
        if not isinstance(value, list):
            return False
        if self.validator is not None:
            for subvalue in value:
                if not self.validator.validate(subvalue):
                    return False
        return True


###############################################################################
## Models
###############################################################################

class Model:
    def __init__(self, collection: Collection):
        self.collection = collection
        self.validator: Optional[Validator[JSON]] = None
        self.defaults: Optional[JSON] = None

    def build_dict(self, **kwargs) -> JSON:
        doc = {}
        for key, value in kwargs.items():
            doc[key] = value
        return doc

    def add(self, **kwargs) -> Document:
        doc = self.build_dict(**kwargs)
        self.d_add(doc)

    def d_add(self, doc) -> Document:
        if self.defaults is not None:
            for key, value in self.defaults.items():
                if key not in doc:
                    if callable(value):
                        doc[key] = value()
                    else:
                        doc[key] = value
        if self.validator is not None and not self.validator.validate(doc):
            return None
        return self.collection.insert_one(doc)

    def get(self, **kwargs) -> Document:
        query = self.build_dict(**kwargs)
        return self.collection.find(query);

    def get_one(self, **kwargs) -> Document:
        query = self.build_dict(**kwargs)
        return self.collection.find_one(query)

    def exists(self, **kwargs) -> Document:
        return self.get_one(**kwargs) is not None


class User(Model):

    def __init__(self, collection: Collection):
        super().__init__(collection)
        self.validator = SequentialValidator([
            ReqFieldsValidator(["username"]),
            PrimaryKeyValidator(collection, ["username"]),
            DictValidator({
                "username": StringValidator(min_size = 4, max_size = 64),
            })
        ])

    def get_user(self, username: str) -> Document:
        return self.get_one(username=username)

    def get_users(self, usernames: List[str]) -> List[Document]:
        return self.collection.find({"username": { "$in": usernames }})

    def add_user(self, username: str) -> Document:
        return self.add(username=username)

    def open_user(self, username: str) -> Document:
        user = self.get_user(username)
        if user:
            return user
        return self.add_user(username)


class World(Model):

    def __init__(self, collection: Collection):
        super().__init__(collection)
        self.validator = SequentialValidator([
            ReqFieldsValidator(["title", "code", "tags", "prerequisites"]),
            PrimaryKeyValidator(collection, ["code"]),
            DictValidator({
                "title": StringValidator(min_size = 1, max_size = 64),
                "code": StringValidator(min_size = 10, max_size = 10),
                "tags": ListValidator(StringValidator(min_size = 4, max_size = 32)),
                "prerequisites": ListValidator(RefKeyValidator(collection, "id")),
                "tests": ListValidator(None)
            })
        ])

class Level(Model):

    def __init__(self, collection: Collection, worlds: Collection):
        super().__init__(collection)
        self.validator = SequentialValidator([
            ReqFieldsValidator(["title", "code", "tags", "prerequisites"]),
            PrimaryKeyValidator(collection, ["code"]),
            DictValidator({
                "title": StringValidator(min_size = 1, max_size = 64),
                "code": StringValidator(min_size = 10, max_size = 10),
                "tags": ListValidator(StringValidator(min_size = 4, max_size = 32)),
                "worldid": RefKeyValidator(worlds, "_id"),
                "prerequisites": ListValidator(RefKeyValidator(collection, "id")),
                "tests": ListValidator(None)
            })
        ])
        self.defaults = {
            "tags": [],
            "prerequisites": [],
            "tests": []
        }

MongoId = str

class Workplace(Model):

    def __init__(self, collection: Collection, users: Collection, levels: Collection):
        super().__init__(collection)
        self.validator = SequentialValidator([
            ReqFieldsValidator(["user", "level", "codeblocks"]),
            PrimaryKeyValidator(collection, ["user", "level"]),
            DictValidator({
                "user": RefKeyValidator(users, "_id"),
                "world": RefKeyValidator(levels, "_id"),
                "codeblocks": ListValidator(StringValidator(min_size=1, max_size=4096))
            })
        ])
        self.defaults = {"codeblocks": []}

    def add_submission(self, userid: MongoId, worldid: MongoId, codeblock):
        doc = self.build_dict(user=userid, world=worldid)

        if self.collection.find_one(doc) is None:
            if self.d_add(doc) is None:
                return None

        document = self.collection.find_one(doc)
        if document is None:
            return None

        code = document["codeblocks"]
        code.append(codeblock)
        if len(code) > MAX_CODEBLOCKS:
            code = code[len(code) - MAX_CODEBLOCKS:]
        return self.collection \
                .find_one_and_update(doc, { '$set': {"codeblocks": code}})

    def get_submissions(self, userids: List[MongoId], worldid: MongoId) -> List[Document]:
        return self.collection.find({ "world": worldid, "user": {"$in": userids}})


    def get_codeblocks(self, userids: List[MongoId], worldid: MongoId) -> Optional[Dict[MongoId, List[str]]]:
        submissions = self.get_submissions(userids, worldid)
        if submissions is None:
            return None
        result = {}
        for submission in submissions:
            result[submission["user"]] = submission["codeblocks"]
        return result







class Solution(Model):

    def __init__(self, collection: Collection, users: Collection, levels: Collection):
        super().__init__(collection)
        self.validator = SequentialValidator([
            ReqFieldsValidator(["user", "level", "mark", "date"]),
            DictValidator({
                "user": RefKeyValidator(users, "_id"),
                "level": RefKeyValidator(levels, "_id"),
                "mark": IntegerValidator(0, 3),
                #TODO: date validator
            })
        ])
        self.defaults = {
            "date": datetime.now
        }

    def add_solution(self, userid, levelid, mark):
        doc = self.build_dict(user=userid, level=levelid, mark=mark)
        self.d_add(doc)

class EllatuDB:

    def __init__(self, host: str, port: int):
        self.client = MongoClient(host, port)
        self.db = self.client[DB_DEV]
        self.user = User(self.db[COL_USERS])
        self.level = Level(self.db[COL_LEVELS])
        self.workplace = Workplace(self.db[COL_WORKPLACES],
                                   self.db[COL_USERS],
                                   self.db[COL_LEVELS])
        self.solution = Solution(self.db[COL_SOLUTIONS],
                                 self.db[COL_USERS],
                                 self.db[COL_LEVELS])


