import os
import joblib

# Absolute path to <project_root>/data_store (three levels up from this file).
DATA_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data_store",
)


class Persistence:
    """Thin wrapper around joblib for saving/loading built artifacts to disk."""

    @staticmethod
    def _path(name: str) -> str:
        # Dataset ids contain "/" (e.g. "beir/quora/test"), which is illegal in filenames.
        safe_name = name.replace("/", "_")
        os.makedirs(DATA_STORE_DIR, exist_ok=True)
        return os.path.join(DATA_STORE_DIR, f"{safe_name}.joblib")

    @staticmethod
    def save(obj, name: str) -> str:
        path = Persistence._path(name)
        joblib.dump(obj, path)
        return path

    @staticmethod
    def load(name: str):
        return joblib.load(Persistence._path(name))

    @staticmethod
    def exists(name: str) -> bool:
        return os.path.exists(Persistence._path(name))
