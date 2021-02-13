"""serializer.py
"""

import jsonpickle
import logging

logger = logging.getLogger(__name__)

class SerializationMethod:
    jspickle = 'jsonpickle'
    # TODO (Can add custom serialization methods)

def jspickle_serialize_fn(path: str, o: object):
    with open(path, 'w') as f:
        f.write(jsonpickle.encode(o))

def jspickle_deserialize_fn(path: str) -> object:
    with open(path, 'r') as f:
        o = jsonpickle.decode(f.read())
    return o

class Serializer:
    def __init__(self, method: str=SerializationMethod.jspickle):
        if method == SerializationMethod.jspickle:
            self.serialize_fn = jspickle_serialize_fn
            self.deserialize_fn = jspickle_deserialize_fn
        else:
            logger.error(f'Method {method} not implemented.')
            raise RuntimeError

    def serialize(self, path: str, o: object):
        self.serialize_fn(path, o)

    def deserialize(self, path: str) -> object:
        return self.deserialize_fn(path)
        
