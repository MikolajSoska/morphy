import typing

import pandas as pd


def flatten_dict(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """
    This method turns nested dictionary into a flat one, merging keys like the following example:
    {
        'a': 1,
        'b': {
            'c': 2,
        },
    }
    is turned into:
    {
        'a': 1,
        'b.c': 2,
    }

    Parameters
    ----------
    data : dict[str, typing.Any]
        Nested dictionary

    Returns
    -------
    dict[str, typing.Any]
        Flat dictionary
    """
    return pd.json_normalize(data).to_dict(orient="records")[0]
