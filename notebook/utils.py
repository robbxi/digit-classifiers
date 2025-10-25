from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np, os

def load_and_split_mnist(folder_path, train_ratio=0.8, seed=42):
    cache_path = f'data/mnist_data_{seed}.npz'

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])
    else:
        return None