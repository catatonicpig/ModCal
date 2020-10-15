

import uncompyle6

with open("uncompiled file.py", "wb") as fileobj:
    uncompyle6.uncompyle_file("plumleecali.cpython-37.pyc", fileobj)
