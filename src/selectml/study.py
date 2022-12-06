""" Some functions for dealing with. """

from typing import Literal

def _check_storage_url(url: str | None) -> str:
    if url is None:
        raise ValueError(f"Invalid URL for database given.")
    return


def create_study(
    storage_url: str | None,
    study_name: str,
    direction: Literal['minimize', 'maximize'],
    load_if_exists: bool = False
) -> None:
    import optuna
    storage_url_ = _check_storage_url(storage_url)
    storage = optuna.storages.get_storage(storage_url_)
    storage_name = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    )
    return storage_name
