import torch
import torch.nn as nn

class HorizonConditionedPhysicsLoss(nn.Module):
    """
    L = sum_h w_h * MSE_h + λ0 * sum_h λh * L_physics_h
    Physics applied more on short horizon, tapered at long horizons.
    """
    def __init__(self, horizon_weights, physics_lambda_by_horizon, lambda0=0.05, physics_fn=None):
        super().__init__()
        self.w = list(horizon_weights)
        self.lh = list(physics_lambda_by_horizon)
        self.lambda0 = float(lambda0)
        self.physics_fn = physics_fn

    def forward(self, y_hat, y, model_out=None, batch=None):
        # Extract tensors from dict outputs
        if isinstance(y_hat, dict):
            y_hat = y_hat.get("flux_pred", y_hat.get("y_hat", y_hat))
        if isinstance(y, dict):
            y = y.get("y_flux", y.get("y", y))
        
        H = y_hat.shape[-1]
        mse = (y_hat - y).pow(2).mean(dim=0)
        loss = sum(self.w[h] * mse[h] for h in range(H))
        
        if self.physics_fn is not None and model_out is not None:
            phy = self.physics_fn(model_out, batch)
            if torch.is_tensor(phy) and phy.ndim == 0:
                loss = loss + self.lambda0 * (sum(self.lh) / max(len(self.lh), 1)) * phy
            elif torch.is_tensor(phy):
                for h in range(min(H, phy.numel())):
                    loss = loss + self.lambda0 * self.lh[h] * phy[h]
        return loss, {"mse": mse.detach().cpu()}