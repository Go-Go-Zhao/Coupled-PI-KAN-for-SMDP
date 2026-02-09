import torch

def solar_field_rhs(T2, T1, F1, I, Ta, params):
    """
    Implements Eq.(1) dT2/dt = f(...)
    params should include: A1,rho,cp1,beta,H,L2,L1,c1,n1
    """
    A1 = params["A1"]
    rho = params["rho"]
    cp1 = params["cp1"]
    beta = params["beta"]
    H = params["H"]
    L2 = params["L2"]
    L1 = params["L1"]
    c1 = params["c1"]

    mes = F1 * rho / c1
    Tbar = 0.5 * (T1 + T2)
    rhs = (1.0 / (A1 * rho * cp1)) * (beta * I - (H / L2) * (Tbar - Ta) - cp1 * mes * (T2 - T1) / L1)
    return rhs

def tank_rhs(T3, T4, T2, T6, V1, m1, m2, Ta, params):
    """
    Implements Eq.(5)(6) for dT3/dt, dT4/dt
    params include: rho,V,cp1,U1,U2
    """
    rho = params["rho"]
    V = params["V"]
    cp1 = params["cp1"]
    U1 = params["U1"]
    U2 = params["U2"]

    dT3 = (m1*T2 + m2*(1 - V1)*T4 - m1*T3 - m2*(1 - V1)*T3 - U1*(T3 - Ta)) / (rho*V*cp1)
    dT4 = (m1*T3 + m2*(1 - V1)*T6 - m1*T4 - m2*(1 - V1)*T4 - U2*(T4 - Ta)) / (rho*V*cp1)
    return dT3, dT4
