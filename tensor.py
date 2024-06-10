import numpy as np
import torch
import torch.nn as nn

# x = np.arange(start=0,stop=40,step=1)
# y = x*3
# # exit(0)

# w = 0.0

# def forward(x):
#     return w*x

# def loss(y,y_pred):
#     return ((y_pred-y)**2).mean()

# def linear_gradient(x,y,y_pred):
#     return 2*np.multiply(x,y_pred-y).mean()

# print(f'Predection before training f(5)= {forward(5):.3f}')

# no_of_iters = 10
# learning_rate = 0.001

# for epoch in range(no_of_iters):
#     y_pred = forward(x)
#     if epoch%1 == 0:
#         print(f"iteration {epoch} - loss:{loss(y,y_pred):.8f} - weight:{w:.3f}")
#     dw = linear_gradient(x,y,y_pred)
#     w -= dw*learning_rate

# print(f'Predection after training f(5)= {forward(5):.3f}')


# x = torch.arange(start=1,end=41,step=1,dtype=torch.float)
# y = x*3
# w = torch.tensor(0.0,dtype=torch.float,requires_grad=True)

# def forward(x):
#     return w*x

# n_iter = 100
# learning_rate = 0.0001
# loss = nn.MSELoss()
# optimizer = torch.optim.SGD([w],lr=learning_rate)

# print(f'Predection before training f(5)= {forward(5):.3f}')

# for epoch in range(n_iter):
#     y_pred = forward(x)
#     l = loss(y,y_pred)
#     l.backward()
#     optimizer.step()
#     if epoch%10 == 0:
#         print(f" epoch {epoch} - loss:{l:.8f} - weight:{w:.3f}")
#     optimizer.zero_grad()

# print(f'Predection after training f(5)= {forward(5):.3f}')




X=torch.tensor([[1],[2],[3],[4],[5]],dtype = torch.float)
Y=torch.tensor([[2],[4],[6],[8],[10]],dtype = torch.float)

x_test = torch.tensor([5],dtype=torch.float)

n_samples,n_features = X.shape
input_params = n_features
output_params = n_features
loss = nn.MSELoss()

class LinearRegresson(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegresson,self).__init__()
        self.lin = nn.Linear(input_dim,output_dim)
    def forward(self,input):
        return self.lin(input)

model = LinearRegresson(n_features,n_features)

print(f"prediction before learning model(5):{model(x_test).item():.3f}")

learning_rate = 0.0001
optimizer  = torch.optim.SGD(model.parameters(),lr=learning_rate)

n_iter  = 100000
for epoch in range(n_iter) :
    y_pred = model(X)
    l = loss(Y,y_pred)
    l.backward()
    optimizer.step()
    if epoch%10000 == 0:
        [w,b] = model.parameters()
        print(f"epoch:{epoch} - weight:{w.item():.8f} - loss:{l:.3f}")
    optimizer.zero_grad()
    
print(f"prediction after learning model(5):{model(x_test).item():.3f}")