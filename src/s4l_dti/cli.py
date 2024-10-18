# Copyright (c) 2024 The Foundation for Research on Information Technologies in Society (IT'IS).
#
# This file is part of s4l-dti
# (see https://github.com/dyollb/s4l-dti).
#
# This software is released under the MIT License.
#  https://opensource.org/licenses/MIT

from __future__ import annotations

import typing
from functools import wraps

if typing.TYPE_CHECKING:
    import typer


def register_command(
    app: typer.Typer,
    func: typing.Callable[..., typing.Any],
    func_name: str | None = None,
):
    """Register function as command"""

    @app.command(name=func_name)
    @wraps(func)
    def foo(*args, **kwargs):
        return func(*args, **kwargs)
