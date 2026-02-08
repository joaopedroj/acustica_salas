
# %%
C_AT5=6.2
L_AT5=9.8
A_AT5=3.6
V_AT5=C_AT5*L_AT5*A_AT5

C_AT9=9.8
L_AT9=7.3
A_AT9=3.4
R_AT9=A_AT9*1.4*0.66
V_AT9=(C_AT9*L_AT9*A_AT9)-R_AT9

C_AT10=6.5
L_AT10=9.7
A_AT10=3.5
R_AT10=A_AT10*1.6*0.83
V_AT10=(C_AT10*L_AT10*A_AT10)-R_AT10

print(f"Volume AT5: {V_AT5:.2f}")
print(f"Volume AT9: {V_AT9:.2f}")
print(f"Volume AT10: {V_AT10:.2f}")
# %%



# import numpy as np

# # Dados da imagem
# dados = [
#     [2.24, 1.96, 1.95, 1.79, 1.67, 1.50, 1.01],
#     [2.22, 1.87, 1.88, 1.75, 1.65, 1.53, 1.03],
#     [2.19, 2.03, 1.97, 1.77, 1.73, 1.45, 1.02],
#     [2.09, 2.14, 1.98, 1.80, 1.65, 1.46, 1.06],
#     [2.35, 2.18, 2.03, 1.78, 1.68, 1.53, 1.07],
#     [2.00, 2.09, 1.90, 1.71, 1.62, 1.49, 1.05],
#     [2.21, 2.10, 1.95, 1.75, 1.70, 1.43, 1.02],
#     [1.99, 2.00, 2.06, 1.73, 1.65, 1.45, 1.05]
# ]
# # Converter para array numpy
# dados_array = np.array(dados)

# # Calcular a média de cada coluna
# medias = np.mean(dados_array, axis=0)

# # Exibir resultados
# print("Média de cada coluna:")
# print("-" * 40)
# for i, media in enumerate(medias, 1):
#     print(f"Coluna {i}: {media:.4f}")

# print("\n" + "=" * 40)
# print(f"Médias: {medias}")
# %%
