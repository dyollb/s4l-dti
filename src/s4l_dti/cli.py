# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-dti
# (see https://github.com/dyollb/s4l-dti).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer


def register_command(
    app: "typer.Typer",
    func: Callable[..., Any],
    func_name: str | None = None,
):
    """Register a function as a Typer CLI command.

    Args:
        app: The Typer application to register the command with.
        func: The callable to expose as a command.
        func_name: Command name override. Defaults to the function name when
            ``None``.
    """

    @app.command(name=func_name)
    @wraps(func)
    def foo(*args, **kwargs):
        return func(*args, **kwargs)
