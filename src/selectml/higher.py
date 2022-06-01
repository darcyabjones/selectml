""" Some higher order functions for dealing with static analysis. """

from typing import cast
from typing import TypeVar, ParamSpec, Concatenate
from typing import Optional
from typing import Callable
from typing import Union
from typing import List
# from typing import Sequence
from typing import Generic


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
P = ParamSpec("P")


def collect_optional(li: List[Optional[T]]) -> Optional[List[T]]:
    if any(lii is None for lii in li):
        return None
    else:
        return cast("List[T]", li)


def ffmap_attr(
    obj: Optional["T"],
    attr: str,
    option: Optional[T],
    *args: P.args,
    **kwargs: P.kwargs
) -> Optional[U]:

    if option is None:
        return None
    elif obj is None:
        return None

    function: Callable[Concatenate[T, P], U] = getattr(obj, attr)
    return function(option, *args, **kwargs)


def safe_attr(cls, attr):
    if cls is None:
        return None
    else:
        return getattr(cls, attr)


def fmap(
    function: Callable[Concatenate[T, P], U],
    option: Optional[T],
    *args: P.args,
    **kwargs: P.kwargs
) -> Optional[U]:
    if option is None:
        return None
    else:
        return function(option, *args, **kwargs)


def fmap2(
    function: Callable[Concatenate[T, Optional[V], P], U],
    option1: Optional[T],
    option2: Optional[V],
    *args: P.args,
    **kwargs: P.kwargs
) -> Optional[U]:
    if option1 is None:
        return None
    elif option2 is None:
        return function(option1, None, *args, **kwargs)
    else:
        return function(option1, option2, *args, **kwargs)


def ffmap(
    function: Optional[Callable[Concatenate[T, P], U]],
    option: Optional[T],
    *args: P.args,
    **kwargs: P.kwargs
) -> Optional[U]:
    if option is None:
        return None
    elif function is None:
        return None
    else:
        return function(option, *args, **kwargs)


def ffmap2(
    function: Optional[Callable[Concatenate[T, Optional[V], P], U]],
    option1: Optional[T],
    option2: Optional[V],
    *args: P.args,
    **kwargs: P.kwargs
) -> Optional[U]:
    if function is None:
        return None
    elif option1 is None:
        return None
    elif option2 is None:
        return function(option1, None, *args, **kwargs)
    else:
        return function(option1, option2, *args, **kwargs)


def applicative(
    function: Callable[[T], Optional[U]],
    option: Optional[T],
) -> Optional[U]:
    """ Same as fmap except for the type signature. """
    if option is None:
        return None
    else:
        return function(option)


def or_else(default: U, option: Optional[T]) -> Union[T, U]:
    """ Replaces None with some default value. """
    if option is None:
        return default
    else:
        return option


class Infix(Generic[T, U, V]):

    def __init__(self, function: Callable[[T, U], V]):
        self.function = function

    def __ror__(self, other: T) -> "InfixPartial[U, V]":
        return InfixPartial(lambda x: self.function(other, x))

    def __or__(self, other: U) -> "InfixPartial[T, V]":
        return InfixPartial(lambda x: self.function(x, other))

    def __call__(self, value1, value2):
        return self.function(value1, value2)


class InfixPartial(Generic[T, V]):

    def __init__(self, function: Callable[[T], V]):
        self.function = function

    def __ror__(self, other: T) -> V:
        return self.function(other)

    def __or__(self, other: T) -> V:
        return self.function(other)

    def __call__(self, value: T) -> V:
        return self.function(value)


class Compose:

    def __ror__(self, f1: Callable[[T], U]) -> "ComposePartial":
        return ComposePartial(function1=f1)

    def __or__(self, f2: Callable[[U], V]) -> "ComposePartial":
        return ComposePartial(function2=f2)

    def __call__(self, f1, f2):
        return lambda x: f1(f2(x))


class ComposePartial:

    def __init__(
        self,
        function1: Optional[Callable[[T], U]] = None,
        function2: Optional[Callable[[U], V]] = None,
    ):
        # One and only one should be none
        assert not all(map(lambda x: x is None, [function1, function2]))
        assert any(map(lambda x: x is None, [function1, function2]))

        self.function1 = function1
        self.function2 = function2
        return

    def __ror__(self, f1: Callable[[T], U]) -> "Callable[[T], V]":
        assert self.function2 is not None
        f2 = cast(Callable[[U], V], self.function2)
        return lambda x: f2(f1(x))

    def __or__(self, f2: Callable[[U], V]) -> "Callable[[T], V]":
        assert self.function1 is not None
        f1 = cast(Callable[[T], U], self.function1)
        return lambda x: f2(f1(x))

    def __call__(
        self,
        function: Union[Callable[[T], U], Callable[[U], V]]
    ) -> "Callable[[T], V]":
        if self.function1 is None:
            f1 = cast(Callable[[T], U], function)

            assert self.function2 is not None
            f2 = cast(Callable[[U], V], self.function2)
            return lambda x: f2(f1(x))

        elif self.function2 is None:
            f2 = cast(Callable[[U], V], function)

            assert self.function1 is not None
            f1 = cast(Callable[[T], U], self.function1)
            return lambda x: f2(f1(x))
        else:
            raise ValueError("Both of the stored functions were None.")


compose = Compose()


class Pipe:

    def __ror__(self, f1: Callable[[U], V]) -> "PipePartial":
        return PipePartial(function1=f1)

    def __or__(self, f2: Callable[[T], U]) -> "PipePartial":
        return PipePartial(function2=f2)

    def __call__(self, f1, f2):
        return lambda x: f1(f2(x))


class PipePartial:

    def __init__(
        self,
        function1: Optional[Callable[[U], V]] = None,
        function2: Optional[Callable[[T], U]] = None,
    ):
        # One and only one should be none
        assert not all(map(lambda x: x is None, [function1, function2]))
        assert any(map(lambda x: x is None, [function1, function2]))

        self.function1 = function1
        self.function2 = function2
        return

    def __ror__(self, f1: Callable[[U], V]) -> "Callable[[T], V]":
        assert self.function2 is not None
        f2 = cast(Callable[[T], U], self.function2)
        return lambda x: f1(f2(x))

    def __or__(self, f2: Callable[[T], U]) -> "Callable[[T], V]":
        assert self.function1 is not None
        f1 = cast(Callable[[U], V], self.function1)
        return lambda x: f1(f2(x))

    def __call__(
        self,
        function: Union[Callable[[T], U], Callable[[U], V]]
    ) -> "Callable[[T], V]":
        if self.function1 is None:
            f1 = cast(Callable[[U], V], function)

            assert self.function2 is not None
            f2 = cast(Callable[[T], U], self.function2)
            return lambda x: f1(f2(x))

        elif self.function2 is None:
            f2 = cast(Callable[[T], U], function)

            assert self.function1 is not None
            f1 = cast(Callable[[U], V], self.function1)
            return lambda x: f1(f2(x))
        else:
            raise ValueError("Both of the stored functions were None.")


pipe = Pipe()


"""
data[]
Maybe[]
Result[]

    fmap
    appl
    unwrap

Compose[]
Pipe[]


(
    do >>
    fn << fmap >> optional
    fmap << fn >> optional
    partial%
    done
)
"""
