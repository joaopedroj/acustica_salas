
import numpy as np
import volume_sala


def sabine(V, T60, S):
    return (0.161*V)/(T60*S)

def sabine_2(V, T60, S_outros, alpha_outros, S_teto):
    return ((0.161*V/T60) - (S_outros*alpha_outros))/S_teto

f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]

# Dados
V_AT9 = 216.03
V_AT10 = 216.03
V_AT5 = 218.74

T60_AT9=[2.16, 2.06, 1.99, 1.75, 1.64, 1.46, 1.01]
T60_AT10=[0.86, 0.58, 0.51, 0.63, 0.71, 0.73, 0.60]

S_AT9 = (2*(volume_sala.A_AT9*volume_sala.C_AT9)) + (2*(volume_sala.A_AT9*volume_sala.L_AT9)) + (2*(volume_sala.L_AT9*volume_sala.C_AT9))
S_outros = (2*(volume_sala.A_AT10*volume_sala.C_AT10)) + (2*(volume_sala.A_AT10*volume_sala.L_AT10)) + (volume_sala.L_AT10*volume_sala.C_AT10) # Área superficial sem considerar o teto
S_teto = (volume_sala.L_AT10*volume_sala.C_AT10)

if __name__ == "__main__":
    for i in range(0, len(T60_AT10)):
        alpha = sabine(V_AT9, T60_AT9[i], S_AT9)
        print(f"alpha {f_central_lista[i]} Hz: {sabine_2(V_AT10, T60_AT10[i], S_outros, alpha, S_teto):.3f}")
        # print(sabine_2(V_AT10, T60_AT10[i], S_outros, alpha, S_teto):.3f)
