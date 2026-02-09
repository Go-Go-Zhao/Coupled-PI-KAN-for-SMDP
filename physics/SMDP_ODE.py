# physics.py
import torch
from config import PARAMS


def solar_field_ode(t, T2, inputs, gradients):
    """
    Eq. 1 in paper: dT2/dt = ... [cite: 141]
    inputs dict: I (irradiance), Ta (ambient), F1 (flow), T1 (inlet approx T4_prev)
    gradients: dT2/dt computed via autograd
    """
    p = PARAMS
    I = inputs['I']
    Ta = inputs['Ta']
    F1 = inputs['F1']
    T1 = inputs['T1']  

    T_bar = (T1 + T2) / 2

    # Eq 2: Equivalent flow rate (simplified conversion)
    m_es = F1 * p['rho'] / 60000.0  # simple unit conversion placeholder

    term1 = p['beta'] * I
    term2 = (p['H'] / p['L2']) * (T_bar - Ta)
    term3 = p['c_p1'] * m_es * ((T2 - T1) / p['L1'])  # Note: Paper has term T2-T1/L1

    rhs = (1.0 / (p['A1'] * p['rho'] * p['c_p1'])) * (term1 - term2 - term3)

    # Residual
    return torch.mean((gradients - rhs) ** 2)


def storage_tank_ode(t, T3, T4, inputs, grads_T3, grads_T4):
    """
    Eq. 5 & 6 in paper [cite: 158-160]
    """
    p = PARAMS
    T2 = inputs['T2_pred']  # From Phi1
    T6 = inputs['T6_prev']  # From previous step/data
    V1 = inputs['V1']
    Ta = inputs['Ta']
    m1 = 0.5  
    m2 = 0.5

    # Eq 5 RHS (T3)
    term_solar_in = m1 * T2
    term_dist_in = m2 * (1 - V1) * T4  # Logic check with paper Eq 5
    term_out = (m1 + m2 * (1 - V1)) * T3
    term_loss = (p['U1'] * (T3 - Ta)) / p['c_p1']

    rhs_T3 = (1.0 / (p['rho'] * p['V'])) * (term_solar_in + term_dist_in - term_out - term_loss)

    # Eq 6 RHS (T4)
    term_upper_in = m1 * T3
    term_ret_in = m2 * (1 - V1) * T6
    term_lower_out = (m1 + m2 * (1 - V1)) * T4
    term_loss_low = (p['U2'] * (T4 - Ta)) / p['c_p1']

    rhs_T4 = (1.0 / (p['rho'] * p['V'])) * (term_upper_in + term_ret_in - term_lower_out - term_loss_low)

    loss_T3 = torch.mean((grads_T3 - rhs_T3) ** 2)
    loss_T4 = torch.mean((grads_T4 - rhs_T4) ** 2)

    return loss_T3 + loss_T4


def compute_gradients(output, input_var):
    """Auto-differentiation wrapper"""
    return torch.autograd.grad(
        output, input_var,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]

