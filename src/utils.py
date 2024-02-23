import joblib as pkl
import os


def pickle(value=None, filename=None):
    if (value and filename) is not None:
        pkl.dump(value=value, filename=filename)
    else:
        ValueError("Pickle is not possible due to missing arguments".capitalize())


def clean_folder(path=None):
    if path is not None:
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

            print("{} - path cleaned".format(path).capitalize())
        else:
            print("{} - path doesn't exist".capitalize())
    else:
        raise ValueError(
            "Clean folder is not possible due to missing arguments".capitalize()
        )
