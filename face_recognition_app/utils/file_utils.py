import os

def delete_representations(path):
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            os.remove(os.path.join(path, file))
