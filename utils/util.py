import os
import json
from functools import wraps
from time import time


def load_json(filepath, warn=True):
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        if warn:
            print(f"{filepath} not found.")
        return dict()

def save_json(file, filepath, **kwargs):
    dump = json.dumps(file, **kwargs)
    with open(filepath, "w") as f:
        f.write(dump)

def safe_name(name):
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    keepcharacters = (' ','.','_')
    return "".join(c for c in name if c.isalnum() or c in keepcharacters).rstrip()

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap
