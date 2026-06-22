#%%
import torch
from torch import pi, sin, cos, sqrt, mean, norm
import matplotlib.pyplot as plt

# def f(x: torch.Tensor, y: torch.Tensor):
#     u = -pi * sin(pi * x) * cos(pi * y)
#     v =  pi * cos(pi * x) * sin(pi * y)
#     return torch.cat([u, v], dim=1)

def f(x: torch.Tensor, y: torch.Tensor):
    vortices = [
        (1.0, -0.45, -0.20, 0.45),
        (-0.7, 0.35, 0.25, 0.35),
        (0.5, 0.10, -0.55, 0.25),
    ]

    u = torch.zeros_like(x)
    v = torch.zeros_like(y)

    for A, x0, y0, s in vortices:
        dx = x - x0
        dy = y - y0
        r2 = dx**2 + dy**2
        gaussian = torch.exp(-r2 / s**2)

        u += -2.0 * A * dy / s**2 * gaussian
        v +=  2.0 * A * dx / s**2 * gaussian

    return torch.cat([u, v], dim=1)


class WindNet2D(torch.nn.Module):
    def __init__(self, hidden_units=20):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, 3),
        )

    def forward(self, xy):
        return self.net(xy)
    
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.data_loss_fn = torch.nn.MSELoss(reduction="mean")

        self.nu = 1e-3

        self.lambda_data = 1.0
        self.lambda_smooth = 0.001
        self.lambda_div = 0.01
        self.lambda_mom = 0.01
        self.lambda_p = 1.0

    def forward(self, uv_pred_meas, uv_meas, w_pred, xy_grid):
        u_pred = w_pred[:, 0]
        v_pred = w_pred[:, 1]
        p_pred = w_pred[:, 2]

        ones = torch.ones_like(u_pred)

        # Columns are derivatives with respect to x and y:
        # grad_u_dxy[:, 0] = du/dx, grad_u_dxy[:, 1] = du/dy
        grad_u_dxy = torch.autograd.grad(u_pred, xy_grid, ones, create_graph=True)[0]
        # grad_v_dxy[:, 0] = dv/dx, grad_v_dxy[:, 1] = dv/dy
        grad_v_dxy = torch.autograd.grad(v_pred, xy_grid, ones, create_graph=True)[0]
        # grad_p_dxy[:, 0] = dp/dx, grad_p_dxy[:, 1] = dp/dy
        grad_p_dxy = torch.autograd.grad(p_pred, xy_grid, ones, create_graph=True)[0]

        u_x = grad_u_dxy[:, 0]
        u_y = grad_u_dxy[:, 1]
        v_x = grad_v_dxy[:, 0]
        v_y = grad_v_dxy[:, 1]
        p_x = grad_p_dxy[:, 0]
        p_y = grad_p_dxy[:, 1]

        grad_ux_dxy = torch.autograd.grad(u_x, xy_grid, ones, create_graph=True)[0]
        grad_uy_dxy = torch.autograd.grad(u_y, xy_grid, ones, create_graph=True)[0]
        grad_vx_dxy = torch.autograd.grad(v_x, xy_grid, ones, create_graph=True)[0]
        grad_vy_dxy = torch.autograd.grad(v_y, xy_grid, ones, create_graph=True)[0]

        u_xx = grad_ux_dxy[:, 0]
        u_yy = grad_uy_dxy[:, 1]
        v_xx = grad_vx_dxy[:, 0]
        v_yy = grad_vy_dxy[:, 1]

        print("Weighted losses:")
        # || uv_meas - uv(samples) ||^2
        data_loss = self.data_loss_fn(uv_pred_meas, uv_meas)
        print(f"data_loss = { self.lambda_data * data_loss}")

        # || grad(u)||^2 != 0
        smooth_loss = mean(grad_u_dxy**2 + grad_v_dxy**2)
        print(f"smoothness_loss = {self.lambda_smooth * smooth_loss}")

        # Navier Stokes mass conservation ||div(u)|| != 0
        div_loss = torch.mean((u_x + v_y)**2)
        print(f"div_loss = {self.lambda_div * div_loss}")
        print("---")

        # Navier Stokes momentum
        mom_x = -self.nu * (u_xx + u_yy) + p_x + u_pred * u_x + v_pred * u_y
        mom_y = -self.nu * (v_xx + v_yy) + p_y + u_pred * v_x + v_pred * v_y
        mom_loss = mean(mom_x**2 + mom_y**2)
        print(f"mom_loss = {self.lambda_mom * mom_loss}")
        print("---")
        
        # Fix the additive pressure offset by enforcing zero mean pressure
        p_loss = mean(p_pred)**2
        print(f"p_loss = {self.lambda_p * p_loss}")
        print("---")

        return (self.lambda_data * data_loss 
                + self.lambda_smooth * smooth_loss
                + self.lambda_div * div_loss
                #+ self.lambda_mom * mom_loss
                # + self.lambda_p * p_loss
                )



if __name__ == "__main__":

    x = torch.linspace(-1.0, 1.0, 24)
    y = torch.linspace(-1.0, 1.0, 24)

    X, Y = torch.meshgrid(x, y, indexing="ij")
    x_flat = X.reshape(-1,1)
    y_flat = Y.reshape(-1,1)

    # Ground Truth - components and speed grid
    uv_gt = f(x_flat, y_flat)
    u_gt = uv_gt[:,0].reshape(X.shape)
    v_gt = uv_gt[:,1].reshape(X.shape)
    C_gt = sqrt(u_gt * u_gt + v_gt * v_gt)

    # Random training set
    n = x_flat.size(0)
    n_samples = 30
    torch.manual_seed(9)
    random_sample_indices = torch.randperm(n)[:n_samples]

    x_sample = x_flat[random_sample_indices]
    y_sample = y_flat[random_sample_indices]
    uv_sample = f(x_sample, y_sample)

    model = WindNet2D(20)
    loss_fn = CustomLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 2e-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


    steps = 500
    loss_array = torch.zeros((steps, 1))

    # Sample points
    sample_input = torch.hstack([x_sample, y_sample])

    # Full grid with differentiable copy for prior evaluation
    grid_input = torch.hstack([x_flat, y_flat])
    prior_input = grid_input.clone().detach().requires_grad_()

    # Training
    for i in range(steps):

        # Send xy coordinate sets through 
        w_pred_data = model(sample_input)
        w_pred_prior = model(prior_input)
        uv_pred_data = w_pred_data[:, :2]
        current_loss = loss_fn(uv_pred_data, uv_sample, w_pred_prior, prior_input)
        loss_array[i] = current_loss.item()
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

    # Prediction for entire grid
    with torch.no_grad():
        w_pred = model(grid_input)

    u_pred = w_pred[:,0].reshape(X.shape)
    v_pred = w_pred[:,1].reshape(X.shape)
    C_pred = sqrt(u_pred * u_pred + v_pred * v_pred) # speeds for colormap (plot)

    fig = plt.figure(figsize=(12,3.5))

    ax_pred = plt.subplot(1, 3, 1)
    ax_true = plt.subplot(1, 3, 2)
    ax_loss = plt.subplot(1, 3, 3)

    norm = plt.Normalize(vmin=min(C_pred.min(), C_gt.min()), 
                        vmax=max(C_pred.max(), C_gt.max()))

    # --- 1. Plot: Prediction ---
    mesh1 = ax_pred.quiver(X, Y, u_pred, v_pred, C_pred, norm=norm)
    fig.colorbar(mesh1, ax=ax_pred, label="Prediction wind speed (m/s)")
    ax_pred.scatter(x_sample.numpy(), y_sample.numpy(), color='red', edgecolor='black')# , label="Training data")
    ax_pred.set_title("Sample-based Prediction")
    ax_pred.set_xlabel("X")
    ax_pred.set_ylabel("Y")
    ax_pred.set_aspect('equal')

    # --- 2. Plot: Ground Truth ---
    mesh2 = ax_true.quiver(X, Y, u_gt, v_gt, C_gt, norm=norm)
    fig.colorbar(mesh2, ax=ax_true)
    ax_true.set_title("Ground truth wind speed (m/s)")
    ax_true.set_aspect('equal')
    ax_true.set_xlabel("X")
    ax_true.set_ylabel("Y")

    # --- 3. Plot: Loss ---
    ax_loss.plot(loss_array.numpy(), color='orange', lw=2)
    ax_loss.set_title("Loss over training steps")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.grid(True, linestyle='--', alpha=0.6)

    # Compute RMSE
    vector_rmse = sqrt(torch.mean((u_pred - u_gt)**2 + (v_pred - v_gt)**2))
    print(f"Vector RMSE = {vector_rmse.item():.4f}")

    plt.tight_layout()
    plt.show()
# %%
