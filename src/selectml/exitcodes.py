#!/usr/bin/env python3

EXIT_VALID = 0
EXIT_KEYBOARD = 1
EXIT_UNKNOWN = 2
EXIT_CLI = 64
EXIT_INPUT_FORMAT = 65
EXIT_INPUT_NOT_FOUND = 66
EXIT_SYSERR = 71
EXIT_CANT_OUTPUT = 73

# EXIT_IOERR = 74


class MyException(Exception):

    def __init__(self, msg: str, ecode: int):
        self.msg = msg
        self.ecode = ecode
        return

    def __str__(self):
        return self.msg

    def __repr__(self):
        return f"{self.__class__}('{self.msg}', {self.ecode})"
