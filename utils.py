from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np, os

def _load_image(file_path):
    img = Image.open(file_path).convert('L')
    return np.array(img).flatten()

def load_and_split_mnist(folder_path, train_ratio=0.8, seed=42,size=1):
    cache_path = f'data/mnist_data_{seed}.npz'

    if os.path.exists(cache_path):
        print("Loading cached data...")
        data = np.load(cache_path)
        return (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])
    
    all_paths, labels = [], []

    for label_name in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_name)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            all_paths.append(os.path.join(label_path, filename))
            labels.append(int(label_name))

    print(f"Found {len(all_paths)} images, loading...")

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(_load_image, all_paths))

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int64)

    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    length = int(len(X)*size)
    n_train = int(train_ratio * length)
    (X_train, y_train), (X_test, y_test) = (X[:n_train], y[:n_train]), (X[n_train:length], y[n_train:length])

    np.savez(f'mnist_data_{seed}.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return (X_train, y_train), (X_test, y_test)
