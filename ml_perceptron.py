import numpy as np
import utils
import torch as torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



(X_train, y_train), (X_test, y_test) = utils.load_and_split_mnist("MNIST")
## Change to [-1,1] for pytorch
X_train = X_train * 2 - 1
X_test = X_test * 2 -1
## Convert from numpy to tensor
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)

X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).long().to(device)




def train(X_train,y_train,learning_rate=.01,batch_size=128,epochs=1):
    model = torch.nn.Sequential(
        torch.nn.Linear(784,256),
        torch.nn.ReLU(),
        torch.nn.Linear(256,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,10)

    ).to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        ## Shuffle data set
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        for i in range(0,len(X_train),batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_pred = model(X_batch)
            loss = loss_func(y_pred,y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch {epoch+1} of {epochs} batch {(i/batch_size)+1} of {len(X_train)/batch_size}')
    model_path = f'multilayer_models/multilayer_{epochs}_epochs_{batch_size}_bs.pth'
    torch.save(model.state_dict(), model_path)
    return model




def make_models(X_train,y_train,learning_rate=.01,batch_size=128,epochs=1):

    model_path = f'multilayer_models/multilayer_{epochs}_epochs_{batch_size}_bs.pth'
    if os.path.exists(model_path):
        model = torch.nn.Sequential(
        torch.nn.Linear(784,256),
        torch.nn.ReLU(),
        torch.nn.Linear(256,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,10)

    )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(device)
    else:
        return train(X_train,y_train,learning_rate=learning_rate,batch_size=batch_size,epochs=epochs)

def multilayer_classifier(model,x_test):
    pred = torch.argmax(model(x_test))
    return pred


def test(model,X_test,y_test,status=True,random=True,n=100):
    correct = 0
    wrong = 0
    for c in range(0,n):
        if random:
            i = np.random.randint(0,n)
        else:
            i = c

        x_test = X_test[i]
        y_truth = y_test[i]
        pred = multilayer_classifier(model,x_test)
        if pred == y_truth:
            correct += 1
        else:
            wrong += 1
        if c % (n/10) == 0 and status:
            print(f"{c}/{n} {100*c/n}% complete")
    
    accuracy = correct / n
    print(f"\nResults for multilayer Classifier Model")
    print(f'{n} {"random" if random else ''} samples tested')
    print(f"{wrong} incorrect, {correct} correct")
    print(f"{accuracy*100}% accuracy\n")

    return


model = make_models(X_train,y_train,learning_rate=.01,batch_size=64,epochs=32)
test(model,X_test,y_test,status=False,random=True,n=5000)