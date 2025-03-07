import dill as pkl
import os

with open(os.path.join(os.path.dirname(__file__),
                       "out.pkl"), "rb") as f:
    res = pkl.load(f)

print(res)