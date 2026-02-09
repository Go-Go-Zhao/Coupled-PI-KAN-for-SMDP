import torch
import torch.nn as nn
from torch.optim import Adam
from models.coupled import CoupledPIKAN
from physics.SMDP_ODE import solar_field_rhs, tank_rhs

def mse(x): return (x**2).mean()

def grad_wrt_t(y, t):
    # y: (B,1) or (B,) ; t: (B,1)
    g = torch.autograd.grad(
        outputs=y,
        inputs=t,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return g

def train_coupled(model, batch, params, epochs=300, stage_epochs=30, tau=0.9, lr=1e-3, wd=1e-3, device="cuda"):
    """
    batch should contain normalized tensors:
      t: (B,1) requires_grad later
      X1: (B,6), y_T2: (B,1)
      X2: (B,7), y_T3T4: (B,2)
      X3: (B,6), y_T5T6T8: (B,3)
    also needed for physics residual:
      For Φ1: T1, F1, I, Ta inside X1; (you can unpack from X1)
      For Φ2: need T2_pred, T6_pred, V1, I, Ta, plus m1,m2
    """
    model.to(device)

    # residual loss weights fixed as 1
    lam_T2 = lam_T3 = lam_T4 = lam_T5 = lam_T6 = lam_T8 = 1.0
    lam1 = torch.tensor(1.0, device=device)  # physics weight for solar
    lam2 = torch.tensor(1.0, device=device)  # physics weight for tank

    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    t = batch["t"].to(device)
    X1 = batch["X1"].to(device)
    yT2 = batch["yT2"].to(device)
    X2 = batch["X2"].to(device)
    yT3T4 = batch["yT3T4"].to(device)
    X3 = batch["X3"].to(device)
    yT5T6T8 = batch["yT5T6T8"].to(device)

    T1 = batch["T1"].to(device)      # (B,1)
    F1 = batch["F1"].to(device)      # (B,1)
    I  = batch["I"].to(device)       # (B,1)
    Ta = batch["Ta"].to(device)      # (B,1)
    V1 = batch["V1"].to(device)      # (B,1)
    m1 = batch["m1"].to(device)      # (B,1) or scalar
    m2 = batch["m2"].to(device)      # (B,1) or scalar

    for ep in range(epochs):
        # 每 stage_epochs 切换训练的子网：Φ1 -> Φ2 -> Φ3 -> 循环
        stage = (ep // stage_epochs) % 3
        if stage == 0:
            model.freeze_except("phi1")
        elif stage == 1:
            model.freeze_except("phi2")
        else:
            model.freeze_except("phi3")

        opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

        opt.zero_grad()

        t_req = t.clone().detach().requires_grad_(True)

        # ---- Φ1: T2 + solar physics ----
        T2_pred = model.forward_phi1(X1)  # (B,1)
        dT2_dt = grad_wrt_t(T2_pred, t_req)

        rhs_T2 = solar_field_rhs(T2_pred, T1, F1, I, Ta, params)
        LFs = mse(dT2_dt - rhs_T2)

        LT2 = mse(T2_pred - yT2)

        # ---- Φ2: (T3,T4) + tank physics ----
        T3T4_pred = model.forward_phi2(X2)  # (B,2)
        T3_pred = T3T4_pred[:, 0:1]
        T4_pred = T3T4_pred[:, 1:2]

        dT3_dt = grad_wrt_t(T3_pred, t_req)
        dT4_dt = grad_wrt_t(T4_pred, t_req)

        # 需要 T6：，用“上一轮/当前Φ3估计”传递
        # 直接用当前Φ3前向得到 T6_pred（冻结时不更新参数）
        T5T6T8_pred = model.forward_phi3(X3)
        T6_pred = T5T6T8_pred[:, 1:2]

        rhs_T3, rhs_T4 = tank_rhs(T3_pred, T4_pred, T2_pred.detach(), T6_pred.detach(), V1, m1, m2, Ta, params)
        LFt = mse(dT3_dt - rhs_T3) + mse(dT4_dt - rhs_T4)

        LT3 = mse(T3_pred - yT3T4[:, 0:1])
        LT4 = mse(T4_pred - yT3T4[:, 1:2])

        # ---- Φ3: (T5,T6,T8) residual only ----
        LT5 = mse(T5T6T8_pred[:, 0:1] - yT5T6T8[:, 0:1])
        LT6 = mse(T5T6T8_pred[:, 1:2] - yT5T6T8[:, 1:2])
        LT8 = mse(T5T6T8_pred[:, 2:3] - yT5T6T8[:, 2:3])

        # total loss
        loss = lam1 * LFs + lam2 * LFt + (lam_T2*LT2 + lam_T3*LT3 + lam_T4*LT4 + lam_T5*LT5 + lam_T6*LT6 + lam_T8*LT8)
        loss.backward()

        # ---- adaptive weight update for lam1, lam2 ----
        with torch.no_grad():
            # 只在训练对应子网时更新对应 physics weight
            if stage == 0:
                # 需要 ∇Φ1 LT2 与 ∇Φ1 LFs 的统计
                lam1 = tau * lam1 + (1 - tau) * lam1  # 保留接口
            if stage == 1:
                lam2 = tau * lam2 + (1 - tau) * lam2

        opt.step()

        if (ep + 1) % 10 == 0:
            print(f"ep {ep+1:4d} stage={stage} loss={loss.item():.4e} LFs={LFs.item():.2e} LFt={LFt.item():.2e}")

    return model
