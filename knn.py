import numpy as np
import utils

(X_train, y_train), (X_test, y_test) = utils.load_and_split_mnist("MNIST")

def compute_distance(X_train,x_test):
    distances = np.sum((X_train - x_test) ** 2, axis=1)
    return distances

def find_knn(distances, y_train, k=3):
    knn_indices = np.argpartition(distances, k)[:k]
    neighbor_labels = y_train[knn_indices]

    ##Count majority

    counts = np.bincount(neighbor_labels)
    pred = np.argmax(counts)

    return pred

def knn(X_train,y_train,x_test, k=3):
    distances = compute_distance(X_train,x_test)
    pred = find_knn(distances,y_train,k)
    return pred

def test(X_train,y_train,X_test,y_test,status=True,random=True,n=100,k=3):
    correct = 0
    wrong = 0
    for c in range(0,n):
        if random:
            i = np.random.randint(0,n)
        else:
            i = c

        x_test = X_test[i]
        y_truth = y_test[i]
        pred = knn(X_train,y_train,x_test, k)
        if pred == y_truth:
            correct += 1
        else:
            wrong += 1
        if c % (n/10) == 0 and status:
            print(f"{c}/{n} {100*c/n}% complete")
    
    accuracy = correct / n
    print(f"\nResults for KNN model with k={k}\n")
    print(f'{n} {"random" if random else ''} samples tested')
    print(f"{wrong} incorrect, {correct} correct")
    print(f"{accuracy*100}% accuracy\n")

    return

test(X_train,y_train,X_test,y_test,status=False,random=True,n=200,k=1)
test(X_train,y_train,X_test,y_test,status=False,random=True,n=200,k=3)
test(X_train,y_train,X_test,y_test,status=False,random=True,n=200,k=5)

