import numpy as np
import utils


(X_train, y_train), (X_test, y_test) = utils.load_and_split_mnist("MNIST")
X_train = (X_train > .5)
X_test = (X_test > .5)

EPSILON = 1e-4



#P(on|class) = P(class|on)P(class)  / P(on)


def naive_bayes(X_train,y_train,x_test):
    total = len(X_train)
    counts = np.zeros((10, X_train.shape[1]), dtype=np.int32)
    class_counts = np.zeros(10)
    for i in range(0 , total):
        c = y_train[i] 
        counts[c] += X_train[i] 
        class_counts[y_train[i]] += 1
        
    probs = np.array(np.log2((class_counts / total)+ EPSILON))
    
    for c in range(0,10):
        for p in range(0,len(x_test)):
            on = x_test[p]
            pixel_count = counts[c][p] + 1
            class_count = class_counts[c] + 2
            p_on_given_class = (pixel_count/class_count)
            p_off_given_class = 1 - p_on_given_class
            probs[c] += (on)*np.log2(p_on_given_class+ EPSILON) + (1-on)*np.log2(p_off_given_class+ EPSILON)

    pred = np.argmax(probs)
    return pred

def test(X_train,y_train,X_test,y_test,status=True,random=True,n=100):
    correct = 0
    wrong = 0
    for c in range(0,n):
        if random:
            i = np.random.randint(0,n)
        else:
            i = c

        x_test = X_test[i]
        y_truth = y_test[i]
        pred = naive_bayes(X_train,y_train,x_test)
        if pred == y_truth:
            correct += 1
        else:
            wrong += 1
        if c % (n/10) == 0 and status:
            print(f"{c}/{n} {100*c/n}% complete")
    
    accuracy = correct / n
    print(f"\nResults for Naive Bayes Model")
    print(f'{n} {"random" if random else ''} samples tested')
    print(f"{wrong} incorrect, {correct} correct")
    print(f"{accuracy*100}% accuracy\n")

    return

test(X_train,y_train,X_test,y_test,status=True,random=False,n=200)