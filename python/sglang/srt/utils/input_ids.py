"""Select the token-list shape before validating its elements."""

from typing import Annotated, Any, Optional, Union

from pydantic import Discriminator, Tag


def _input_ids_shape(value: Any) -> str:
    # Inspect only the outer structure. Pydantic still validates every element
    # in the selected branch, including mixed or malformed lists.
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return "batch"
    return "single"


InputIds = Optional[
    Annotated[
        Union[
            Annotated[list[int], Tag("single")],
            Annotated[list[list[int]], Tag("batch")],
        ],
        Discriminator(_input_ids_shape),
    ]
]
