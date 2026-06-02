import os
import requests
import pandas as pd

from config import (
    EIA_REGION_DATA_URL,
    EIA_RESPONDENT,
    EIA_TYPES,
    EIA_DEFAULT_LENGTH,
)


def get_eia_api_key():
    """
    Read EIA API key from environment variable.
    Do not hardcode the API key in source code.
    """
    return os.getenv("EIA_API_KEY")


def fetch_eia_region_data(
    api_key,
    respondent=EIA_RESPONDENT,
    start=None,
    end=None,
    length=EIA_DEFAULT_LENGTH,
):
    """
    Fetch hourly region-level EIA data for a balancing authority / region.

    For CAL dashboard, this fetches:
    - D  : actual demand
    - DF : demand forecast
    """
    if not api_key:
        raise ValueError(
            "EIA API key is missing. Please set EIA_API_KEY as an environment variable."
        )

    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": length,
    }

    for eia_type in EIA_TYPES:
        params.setdefault("facets[type][]", [])
        params["facets[type][]"].append(eia_type)

    if start is not None:
        params["start"] = start

    if end is not None:
        params["end"] = end

    response = requests.get(
        EIA_REGION_DATA_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    json_data = response.json()

    data = json_data.get("response", {}).get("data", [])

    df = pd.DataFrame(data)

    return df