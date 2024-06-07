import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Neural Network Model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

# PDE Residual
def pde_residual(net, x, y, k=50):
    T = net(torch.cat([x, y], dim=1))
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
    return k * (T_xx + T_yy)

# Boundary Conditions
def boundary_conditions(net, x, y, T_bc):
    T = net(torch.cat([x, y], dim=1))
    return T - T_bc

# Loss Function
def loss_function(pde_res, bc_res):
    pde_loss = torch.mean(pde_res**2)
    bc_loss = torch.mean(bc_res**2)
    return pde_loss + bc_loss

# Generate Data Points (Boundary and Collocation Points)
N_T = 10000  # Number of datapoints
N_F = 10000  # Number of collocation points
domain_size = 100e-6  # 100 micrometers

# Boundary Points
x_left = np.zeros((N_T//4, 1), dtype=float)
y_left = np.random.uniform(0, domain_size, (N_T//4, 1))
X_left = np.hstack((x_left, y_left))

x_top = np.random.uniform(0, domain_size, (N_T//4, 1))
y_top = np.full((N_T//4, 1), domain_size, dtype=float)
X_top = np.hstack((x_top, y_top))

x_bottom = np.random.uniform(0, domain_size, (N_T//4, 1))
y_bottom = np.zeros((N_T//4 , 1), dtype=float)
X_bottom = np.hstack((x_bottom, y_bottom))

x_right = np.full((N_T//4, 1), domain_size, dtype=float)
y_right = np.random.uniform(0, domain_size, (N_T//4, 1))
X_right = np.hstack((x_right, y_right))

X_T_train = np.vstack((X_left, X_top, X_bottom, X_right))

# Shuffle X_T_train
index = np.arange(0, N_T)
np.random.shuffle(index)
X_T_train = X_T_train[index, :]

# Collocation Points
X_F_train = np.zeros((N_F, 2), dtype=float)
for row in range(N_F):
    x = np.random.uniform(0, domain_size)
    y = np.random.uniform(0, domain_size)
    X_F_train[row, 0] = x
    X_F_train[row, 1] = y

# Add the boundary points to collocation points
X_F_train = np.vstack((X_F_train, X_T_train))

# Convert to Torch tensors
X_F_train = torch.tensor(X_F_train, dtype=torch.float32, requires_grad=True)
X_T_train = torch.tensor(X_T_train, dtype=torch.float32)

# Initialize the neural network
layers = [2, 20, 20, 20, 20, 1]
net = PINN(layers)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Training Loop
epochs = 1000
T_bc = torch.zeros((N_T, 1), dtype=torch.float32)  # Assuming T=0 at the boundaries for simplicity

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Split collocation points into x and y
    x_F = X_F_train[:, 0].view(-1, 1)
    y_F = X_F_train[:, 1].view(-1, 1)
    
    # Compute the PDE residuals
    pde_res = pde_residual(net, x_F, y_F)
    
    # Split boundary points into x and y
    x_T = X_T_train[:, 0].view(-1, 1)
    y_T = X_T_train[:, 1].view(-1, 1)
    
    # Compute the boundary condition residuals
    bc_res = boundary_conditions(net, x_T, y_T, T_bc)
    
    # Compute the loss
    loss = loss_function(pde_res, bc_res)
    
    # Backpropagation and optimization
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluate Temperature Distribution
def evaluate_temperature(net, x, y):
    with torch.no_grad():
        T = net(torch.cat([x, y], dim=1))
    return T

# Generate a grid of points
x = np.linspace(0, domain_size, 100)
y = np.linspace(0, domain_size, 100)
X, Y = np.meshgrid(x, y)
x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1)

# Compute the temperature at the grid points
net.eval()  # Set the network to evaluation mode
T = evaluate_temperature(net, x_tensor, y_tensor)

# Convert temperature to a numpy array
T = T.numpy().reshape(100, 100)

# Plot the temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, T, 20, cmap='jet')
plt.colorbar(label='Temperature')
plt.title('Temperature Distribution')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# Compute and Plot Flux Heat Map
def compute_flux(net, x, y, k=50):
    x.requires_grad_(True)
    y.requires_grad_(True)
    T = net(torch.cat([x, y], dim=1))
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    q_x = -k * T_x
    q_y = -k * T_y
    return q_x, q_y

# Compute the flux at the grid points
q_x, q_y = compute_flux(net, x_tensor, y_tensor)

# Convert flux components to numpy arrays
q_x = q_x.detach().numpy().reshape(100, 100)
q_y = q_y.detach().numpy().reshape(100, 100)
flux_magnitude = np.sqrt(q_x**2 + q_y**2)

# Plot the heat map of the flux magnitude
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, flux_magnitude, 20, cmap='hot')
plt.colorbar(label='Flux Magnitude')
plt.title('Heat Map of Flux Magnitude')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()
