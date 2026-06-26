
from typing import Annotated

from fastapi import Depends


def get_current_user() -> None:
    """
    Placeholder dependency.

    Returns:
        None

    Notes:
        Authentication is not enabled in the current application version.
    """

    return None


CurrentUser = Annotated[
    None,
    Depends(get_current_user),
]


"""
V2 roadmap

- OAuth2PasswordBearer
- JWT validation
- Refresh tokens
- Role-based authorization
- User ownership validation
"""