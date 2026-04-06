from pathlib import Path

from joblib import dump, load

# Deux fonctions simples :
# save_model(model, path: Path) -> None
# load_model(path: Path) -> object
# Crée le dossier si besoin : path.parent.mkdir(parents=True, exist_ok=True)


def save_model(model: object, path: Path) -> None:
    """Save a trained model to disk using joblib.
    Args:
        model (object): The trained model to save.
        path (Path): The file path where the model will be saved.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    print(" File saved at : ", path)


def load_model(path: Path) -> object:
    """Load a trained model from disk using joblib.
    Args:
        path (Path): The file path from where the model will be loaded.
    Returns:
        object: The loaded model.
    """
    if not path.exists():
        raise FileNotFoundError("The file doesn't exist")

    print(" File loaded from : ", path)
    return load(path)
