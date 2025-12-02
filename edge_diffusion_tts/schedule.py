"""
Diffusion schedule module.

Implements DDPM and DDIM scheduling with support for progressive distillation.
"""

import torch
import torch.nn.functional as F


class DiffusionSchedule:
    """
    Improved diffusion schedule with support for:
    - Standard DDPM
    - DDIM (deterministic sampling)
    - Progressive distillation
    - Consistency models
    
    Args:
        T: Number of diffusion timesteps
        beta_start: Starting beta value for linear schedule
        beta_end: Ending beta value for linear schedule
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        T: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu"
    ):
        self.T = T
        self.device = device
        
        # Cosine schedule
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps, device=device)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        
        # Precompute useful quantities
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_alpha_bar_minus_one = torch.sqrt(1.0 / self.alpha_bar - 1)
        
        # For DDPM sampling
        alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        
        # Lambda (log-SNR) for DPM-Solver
        self.lambda_t = torch.log(self.sqrt_alpha_bar / self.sqrt_one_minus_alpha_bar)
    
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x0: Clean data [B, T, D]
            t: Timesteps [B]
            noise: Optional pre-sampled noise [B, T, D]
        
        Returns:
            x_t: Noisy data at timestep t
            noise: The noise added
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise
    
    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct x_0 from x_t and predicted noise.
        
        Args:
            x_t: Noisy data [B, T, D]
            t: Timesteps [B]
            eps: Predicted noise [B, T, D]
        
        Returns:
            x0_pred: Predicted clean data
        """
        sqrt_recip = self.sqrt_recip_alpha_bar[t][:, None, None]
        sqrt_recip_m1 = self.sqrt_recip_alpha_bar_minus_one[t][:, None, None]
        return sqrt_recip * x_t - sqrt_recip_m1 * eps
    
    def predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct x_0 from x_t and predicted velocity (v-prediction).
        
        v = sqrt(ᾱ)*ε - sqrt(1-ᾱ)*x0
        x0 = sqrt(ᾱ)*x_t - sqrt(1-ᾱ)*v
        
        More numerically stable than ε-prediction at high noise levels.
        """
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sqrt_ab * x_t - sqrt_1mab * v
    
    def predict_eps_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert v-prediction to ε-prediction.
        
        ε = sqrt(1-ᾱ)*x_t + sqrt(ᾱ)*v
        """
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sqrt_1mab * x_t + sqrt_ab * v
    
    def get_v_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute v-prediction target for training.
        
        v = sqrt(ᾱ)*ε - sqrt(1-ᾱ)*x0
        """
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sqrt_ab * noise - sqrt_1mab * x0
    
    def get_ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eps_pred: torch.Tensor,
        eta: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM sampling step - deterministic when eta=0.
        
        Args:
            x_t: Current noisy data [B, T, D]
            t: Current timesteps [B]
            t_prev: Previous timesteps [B]
            eps_pred: Predicted noise [B, T, D]
            eta: Stochasticity parameter (0 = deterministic)
        
        Returns:
            x_prev: Data at previous timestep
            x0_pred: Predicted clean data
        """
        alpha_bar_t = self.alpha_bar[t][:, None, None]
        
        # Handle t_prev = -1 (final step)
        alpha_bar_t_prev = torch.where(
            t_prev.unsqueeze(-1).unsqueeze(-1) >= 0,
            self.alpha_bar[t_prev.clamp(min=0)][:, None, None],
            torch.ones_like(alpha_bar_t)
        )
        
        # Predict x_0
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -3, 3)  # Stability clipping
        
        # Direction pointing to x_t
        sigma = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * eps_pred
        
        # Sample x_{t-1}
        noise = torch.randn_like(x_t) if eta > 0 else 0
        x_prev = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt + sigma * noise
        
        return x_prev, x0_pred
    
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard DDPM sampling step.
        
        Args:
            x_t: Current noisy data [B, T, D]
            t: Current timesteps [B]
            eps_pred: Predicted noise [B, T, D]
        
        Returns:
            x_prev: Data at previous timestep
        """
        alpha = self.alphas[t][:, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None]
        beta = self.betas[t][:, None, None]
        
        # Mean
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = beta / torch.sqrt(1.0 - alpha_bar)
        mean = coef1 * (x_t - coef2 * eps_pred)
        
        # Variance
        var = self.posterior_variance[t][:, None, None]
        noise = torch.randn_like(x_t)
        
        # Don't add noise at t=0
        nonzero_mask = (t > 0).float()[:, None, None]
        x_prev = mean + nonzero_mask * torch.sqrt(var) * noise
        
        return x_prev
    
    def get_schedule_for_steps(self, num_steps: int) -> list[int]:
        """
        Get evenly spaced timesteps for reduced-step sampling.
        
        Args:
            num_steps: Number of sampling steps
        
        Returns:
            List of timesteps to use
        """
        stride = self.T // num_steps
        timesteps = list(range(self.T - 1, 0, -stride))[:num_steps]
        return timesteps
    
    def to(self, device: str) -> "DiffusionSchedule":
        """Move schedule to device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_recip_alpha_bar = self.sqrt_recip_alpha_bar.to(device)
        self.sqrt_recip_alpha_bar_minus_one = self.sqrt_recip_alpha_bar_minus_one.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.lambda_t = self.lambda_t.to(device)
        return self


class DPMSolverPP:
    """
    DPM-Solver++ for fast, high-quality diffusion sampling.
    
    Based on "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion 
    Probabilistic Models" (Lu et al., 2022).
    
    Key features:
    - Uses log-SNR (lambda) parameterization for optimal timestep spacing
    - Supports 1st, 2nd, and 3rd order solvers
    - Works with v-prediction models
    - 5-15 steps for high quality (vs 50+ for DDIM)
    
    Args:
        schedule: DiffusionSchedule instance
        order: Solver order (1, 2, or 3). Higher = better but needs history.
        predict_x0: If True, model predicts x0 directly. If False, predicts v.
    """
    
    def __init__(
        self,
        schedule: DiffusionSchedule,
        order: int = 2,
        predict_x0: bool = False
    ):
        self.schedule = schedule
        self.order = order
        self.predict_x0 = predict_x0
        self.device = schedule.device
    
    def get_time_steps(self, num_steps: int, max_t: int = None) -> torch.Tensor:
        """
        Get timesteps with logarithmic spacing in lambda space.
        
        This is better than linear spacing because it allocates more
        steps where the signal-to-noise ratio changes fastest.
        """
        max_t = max_t or (self.schedule.T - 1)
        
        # Get lambda bounds
        lambda_max = self.schedule.lambda_t[1].item()  # t=1 (most signal)
        lambda_min = self.schedule.lambda_t[max_t].item()  # t=max (most noise)
        
        # Logarithmic spacing in lambda
        lambdas = torch.linspace(lambda_min, lambda_max, num_steps + 1, device=self.device)
        
        # Convert lambda back to timestep indices
        timesteps = []
        for lam in lambdas[:-1]:  # Skip last (t=0)
            # Find closest t
            diffs = (self.schedule.lambda_t - lam).abs()
            t = diffs.argmin().item()
            t = max(1, min(t, max_t))
            timesteps.append(t)
        
        return torch.tensor(timesteps, device=self.device, dtype=torch.long)
    
    def model_to_x0(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Convert model output to x0 prediction."""
        if self.predict_x0:
            return model_output
        else:
            # v-prediction: x0 = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v
            return self.schedule.predict_x0_from_v(x_t, t, model_output)
    
    def first_order_update(
        self,
        x: torch.Tensor,
        x0_pred: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        First-order DPM-Solver update (similar to DDIM but better parameterized).
        
        x_{t_prev} = (sigma_{t_prev}/sigma_t) * x_t + alpha_{t_prev} * (1 - e^{lambda_t - lambda_{t_prev}}) * x0_pred
        """
        alpha_t = self.schedule.sqrt_alpha_bar[t][:, None, None]
        alpha_prev = self.schedule.sqrt_alpha_bar[t_prev][:, None, None]
        sigma_t = self.schedule.sqrt_one_minus_alpha_bar[t][:, None, None]
        sigma_prev = self.schedule.sqrt_one_minus_alpha_bar[t_prev][:, None, None]
        
        lambda_t = self.schedule.lambda_t[t][:, None, None]
        lambda_prev = self.schedule.lambda_t[t_prev][:, None, None]
        
        h = lambda_prev - lambda_t  # Step size in lambda space
        
        # DPM-Solver first-order update
        x_prev = (sigma_prev / sigma_t) * x + alpha_prev * (1 - torch.exp(-h)) * x0_pred
        
        return x_prev
    
    def second_order_update(
        self,
        x: torch.Tensor,
        x0_pred: torch.Tensor,
        x0_prev: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        t_prev2: torch.Tensor
    ) -> torch.Tensor:
        """
        Second-order DPM-Solver++ update (uses one previous prediction).
        
        More accurate than first-order, especially for fewer steps.
        """
        alpha_prev = self.schedule.sqrt_alpha_bar[t_prev][:, None, None]
        sigma_t = self.schedule.sqrt_one_minus_alpha_bar[t][:, None, None]
        sigma_prev = self.schedule.sqrt_one_minus_alpha_bar[t_prev][:, None, None]
        
        lambda_t = self.schedule.lambda_t[t][:, None, None]
        lambda_prev = self.schedule.lambda_t[t_prev][:, None, None]
        lambda_prev2 = self.schedule.lambda_t[t_prev2][:, None, None]
        
        h = lambda_prev - lambda_t
        h_prev = lambda_prev2 - lambda_prev
        r = h_prev / h
        
        # Second-order correction
        D0 = x0_pred
        D1 = (1 / r) * (x0_pred - x0_prev)
        
        x_prev = (
            (sigma_prev / sigma_t) * x 
            + alpha_prev * (1 - torch.exp(-h)) * D0
            + alpha_prev * ((1 - torch.exp(-h)) / h + 1) * D1 * 0.5
        )
        
        return x_prev
    
    def third_order_update(
        self,
        x: torch.Tensor,
        x0_preds: list,  # [x0_pred, x0_prev, x0_prev2]
        t: torch.Tensor,
        t_prev: torch.Tensor,
        ts_history: list  # [t_prev2, t_prev3]
    ) -> torch.Tensor:
        """
        Third-order DPM-Solver++ update (uses two previous predictions).
        
        Most accurate, but needs 2 steps of history.
        """
        alpha_prev = self.schedule.sqrt_alpha_bar[t_prev][:, None, None]
        sigma_t = self.schedule.sqrt_one_minus_alpha_bar[t][:, None, None]
        sigma_prev = self.schedule.sqrt_one_minus_alpha_bar[t_prev][:, None, None]
        
        lambda_t = self.schedule.lambda_t[t][:, None, None]
        lambda_prev = self.schedule.lambda_t[t_prev][:, None, None]
        
        h = lambda_prev - lambda_t
        
        # Use history for higher-order correction
        D0 = x0_preds[0]
        D1 = x0_preds[0] - x0_preds[1]
        D2 = x0_preds[0] - 2 * x0_preds[1] + x0_preds[2]
        
        x_prev = (
            (sigma_prev / sigma_t) * x
            + alpha_prev * (1 - torch.exp(-h)) * D0
            + alpha_prev * ((1 - torch.exp(-h)) / h + 1) * D1 * 0.5
            + alpha_prev * ((1 - torch.exp(-h)) / (h ** 2) + 0.5 / h + 0.5) * D2 / 6
        )
        
        return x_prev
    
    @torch.no_grad()
    def sample(
        self,
        model,
        x_T: torch.Tensor,
        sem_features: torch.Tensor,
        num_steps: int = 10,
        max_t: int = None,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Sample from the diffusion model using DPM-Solver++.
        
        Args:
            model: Diffusion model (decoder)
            x_T: Initial noise [B, T, D]
            sem_features: Semantic conditioning [B, S, D]
            num_steps: Number of sampling steps (5-20 recommended)
            max_t: Maximum timestep to start from
            return_intermediates: If True, return all intermediate x0 predictions
        
        Returns:
            x_0: Clean sample [B, T, D]
        """
        max_t = max_t or 950  # Avoid very high noise
        
        # Get logarithmically-spaced timesteps
        timesteps = self.get_time_steps(num_steps, max_t)
        
        x = x_T
        x0_history = []
        t_history = []
        intermediates = []
        
        B = x.shape[0]
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t.item(), device=self.device, dtype=torch.long)
            step_idx = torch.full((B,), i, device=self.device, dtype=torch.long)
            
            # Get model prediction
            model_output = model(x, t_tensor, sem_features=sem_features, step_idx=step_idx)
            
            # Align sequence lengths
            min_len = min(model_output.shape[1], x.shape[1])
            model_output = model_output[:, :min_len, :]
            x = x[:, :min_len, :]
            
            # Convert to x0 prediction
            x0_pred = self.model_to_x0(model_output, x, t_tensor)
            x0_pred = torch.clamp(x0_pred, -3, 3)  # Stability clipping
            
            if return_intermediates:
                intermediates.append(x0_pred.clone())
            
            # Get next timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = torch.tensor(0, device=self.device)
            t_prev_tensor = torch.full((B,), t_prev.item(), device=self.device, dtype=torch.long)
            
            # Choose update order based on history available
            if self.order == 1 or len(x0_history) == 0:
                x = self.first_order_update(x, x0_pred, t_tensor, t_prev_tensor)
            elif self.order == 2 or len(x0_history) == 1:
                x = self.second_order_update(
                    x, x0_pred, x0_history[-1],
                    t_tensor, t_prev_tensor, t_history[-1]
                )
            else:  # order >= 3 and enough history
                x = self.third_order_update(
                    x, [x0_pred] + x0_history[-2:],
                    t_tensor, t_prev_tensor, t_history[-2:]
                )
            
            # Update history
            x0_history.append(x0_pred)
            t_history.append(t_prev_tensor)
            
            # Keep only what we need
            if len(x0_history) > 2:
                x0_history.pop(0)
                t_history.pop(0)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def to(self, device: str) -> "DPMSolverPP":
        """Move solver to device."""
        self.device = device
        self.schedule = self.schedule.to(device)
        return self

