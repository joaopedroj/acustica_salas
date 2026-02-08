# %%
# T60

import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 0.8

# Dados de T60 médio para cada sala
frequencias = [125, 250, 500, 1000, 2000, 4000, 8000]
frequencias_labels = ['125', '250', '500', '1000', '2000', '4000', '8000']

# Dados das tabelas
at5 = [1.16, 1.04, 0.73, 0.65, 0.55, 0.63, 0.56]
at9_config1 = [2.16, 2.06, 2.00, 1.75, 1.64, 1.46, 1.01]
at9_config2 = [2.16, 2.05, 1.97, 1.76, 1.67, 1.48, 1.04]
at10 = [0.86, 0.58, 0.51, 0.63, 0.70, 0.73, 0.60]

# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar linhas com marcadores
ax.plot(frequencias, at5, marker='o', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT5', color='#1976D2')

ax.plot(frequencias, at9_config1, marker='s', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 1', color="#EC9B44")

ax.plot(frequencias, at9_config2, marker='^', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 2', color="#65BEEE")

ax.plot(frequencias, at10, marker='D', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT10', color='#D32F2F' )

# Configurar eixos
ax.set_xlabel('Frequência [Hz]', fontsize=18, fontweight='normal')
ax.set_ylabel('T$_{60}$ [s]', fontsize=18, fontweight='normal')

# Configurar escala logarítmica no eixo x
ax.set_xscale('log')
ax.set_xticks(frequencias)
ax.set_xticklabels(frequencias_labels)

# Configurar limites do eixo y
ax.set_ylim(0, 2.5)
ax.set_yticks(np.arange(0, 2.6, 0.5))

# Grade
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legenda
ax.legend(loc='upper right', frameon=True, fontsize=15, 
          edgecolor='black', fancybox=False, shadow=False)

# Ajustar layout
plt.tight_layout()

# Salvar figura
plt.savefig('T60_comparacao_apre.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Exibir
plt.show()

print("Gráfico gerado com sucesso!")
# print("\nResumo dos dados:")
# print(f"AT5: T60 varia de {min(at5):.2f}s a {max(at5):.2f}s")
# print(f"AT9 Config.1: T60 varia de {min(at9_config1):.2f}s a {max(at9_config1):.2f}s")
# print(f"AT9 Config.2: T60 varia de {min(at9_config2):.2f}s a {max(at9_config2):.2f}s")
# print(f"AT10: T60 varia de {min(at10):.2f}s a {max(at10):.2f}s")

# %%

############################################################################################################
# %%
# C50


import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 0.8

# Dados de T60 médio para cada sala
frequencias = [125, 250, 500, 1000, 2000, 4000, 8000]
frequencias_labels = ['125', '250', '500', '1000', '2000', '4000', '8000']

# Dados das tabelas
at5 = [-6.06, -1.03, -1.49, 0.49, 0.82, 1.01, 2.92]
at9_config1 = [-6.82, -7.48, -7.07, -5.64, -5.09, -3.99, -0.81]
at9_config2 = [-9.87, -7.11, -7.02, -5.24, -4.69, -3.60, -0.42]
at10 = [-2.98, -1.54, 1.71, 1.38, 2.99, 1.68, 3.62]

# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar linhas com marcadores
ax.plot(frequencias, at5, marker='o', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT5', color='#1976D2')

ax.plot(frequencias, at9_config1, marker='s', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 1', color="#EC9B44")

ax.plot(frequencias, at9_config2, marker='^', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 2', color="#65BEEE")

ax.plot(frequencias, at10, marker='D', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT10', color='#D32F2F')

# Configurar eixos
ax.set_xlabel('Frequência [Hz]', fontsize=18, fontweight='normal')
ax.set_ylabel('C$_{50}$ [dB]', fontsize=18, fontweight='normal')

# Configurar escala logarítmica no eixo x
ax.set_xscale('log')
ax.set_xticks(frequencias)
ax.set_xticklabels(frequencias_labels)

# Configurar limites do eixo y
ax.set_ylim(-10.5, 7.5)
ax.set_yticks(np.arange(-10.0, 8.0, 1.0))

# Grade
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# # Legenda
# ax.legend(loc='upper right', frameon=True, fontsize=10, 
#           edgecolor='black', fancybox=False, shadow=False)


# Legenda
ax.legend(loc='upper left', frameon=True, fontsize=15, 
          edgecolor='black', fancybox=False, shadow=False)

# Ajustar layout
plt.tight_layout()

# Salvar figura
plt.savefig('C50_comparacao_apre.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Exibir
plt.show()

# print("\nResumo dos dados:")
# print(f"AT5: T60 varia de {min(at5):.2f}s a {max(at5):.2f}s")
# print(f"AT9 Config.1: T60 varia de {min(at9_config1):.2f}s a {max(at9_config1):.2f}s")
# print(f"AT9 Config.2: T60 varia de {min(at9_config2):.2f}s a {max(at9_config2):.2f}s")
# print(f"AT10: T60 varia de {min(at10):.2f}s a {max(at10):.2f}s")
# %%


############################################################################################################
# %%
# D50


import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 0.8

# Dados de T60 médio para cada sala
frequencias = [125, 250, 500, 1000, 2000, 4000, 8000]
frequencias_labels = ['125', '250', '500', '1000', '2000', '4000', '8000']

# Dados das tabelas
at5 = [0.24, 0.45, 0.42, 0.53, 0.55, 0.55, 0.66]
at9_config1 = [0.19, 0.16, 0.17, 0.22, 0.24, 0.29, 0.45]
at9_config2 = [0.10, 0.17, 0.17, 0.24, 0.26, 0.31, 0.48]
at10 = [0.35, 0.42, 0.59, 0.58, 0.66, 0.59, 0.69]

# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar linhas com marcadores
ax.plot(frequencias, at5, marker='o', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT5', color='#1976D2')

ax.plot(frequencias, at9_config1, marker='s', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 1', color="#EC9B44")

ax.plot(frequencias, at9_config2, marker='^', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 2', color="#65BEEE")

ax.plot(frequencias, at10, marker='D', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT10', color='#D32F2F')

# Configurar eixos
ax.set_xlabel('Frequência [Hz]', fontsize=18, fontweight='normal')
ax.set_ylabel('D$_{50}$', fontsize=18, fontweight='normal')

# Configurar escala logarítmica no eixo x
ax.set_xscale('log')
ax.set_xticks(frequencias)
ax.set_xticklabels(frequencias_labels)

# Configurar limites do eixo y
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.1, 0.1))

# Grade
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legenda
ax.legend(loc='upper right', frameon=True, fontsize=15, 
          edgecolor='black', fancybox=False, shadow=False)

# Ajustar layout
plt.tight_layout()

# Salvar figura
plt.savefig('D50_comparacao_apre.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Exibir
plt.show()

# print("\nResumo dos dados:")
# print(f"AT5: T60 varia de {min(at5):.2f}s a {max(at5):.2f}s")
# print(f"AT9 Config.1: T60 varia de {min(at9_config1):.2f}s a {max(at9_config1):.2f}s")
# print(f"AT9 Config.2: T60 varia de {min(at9_config2):.2f}s a {max(at9_config2):.2f}s")
# print(f"AT10: T60 varia de {min(at10):.2f}s a {max(at10):.2f}s")
# %%

############################################################################################################
# %%
# SNR sem ventiladores


import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 0.8

# Dados de T60 médio para cada sala
frequencias = [125, 250, 500, 1000, 2000, 4000, 8000]
frequencias_labels = ['125', '250', '500', '1000', '2000', '4000', '8000']

# Dados das tabelas
at5 = [19.82, 21.15, 21.46, 15.09, 23.38, 32.96, 17.25]
at9 = [23.99, 28.44, 29.76, 32.33, 30.38, 28.40, 23.99]
at10 = [27.10, 28.37, 29.80, 26.21, 35.04, 38.87, 20.13]

# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar linhas com marcadores
ax.plot(frequencias, at5, marker='o', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT5', color='#1976D2')

ax.plot(frequencias, at9, marker='s', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9', color="#EC9B44")

# ax.plot(frequencias, at9_config2, marker='^', linestyle='-', linewidth=1.5, 
#         markersize=6, label='AT9 - Config. 2', color="#65BEEE")

ax.plot(frequencias, at10, marker='D', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT10', color='#D32F2F')

# Configurar eixos
ax.set_xlabel('Frequência [Hz]', fontsize=18, fontweight='normal')
ax.set_ylabel('SNR [dB]', fontsize=18, fontweight='normal')

# Configurar escala logarítmica no eixo x
ax.set_xscale('log')
ax.set_xticks(frequencias)
ax.set_xticklabels(frequencias_labels)

# Configurar limites do eixo y
ax.set_ylim(00.0, 40.0)
ax.set_yticks(np.arange(0, 42, 2.0))

# Grade
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legenda
ax.legend(loc='upper left', frameon=True, fontsize=15, 
          edgecolor='black', fancybox=False, shadow=False)

# Ajustar layout
plt.tight_layout()

# Salvar figura
plt.savefig('SNR_SV_comparacao_2_apre.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Exibir
plt.show()

# print("\nResumo dos dados:")
# print(f"AT5: T60 varia de {min(at5):.2f}s a {max(at5):.2f}s")
# print(f"AT9 Config.1: T60 varia de {min(at9_config1):.2f}s a {max(at9_config1):.2f}s")
# print(f"AT9 Config.2: T60 varia de {min(at9_config2):.2f}s a {max(at9_config2):.2f}s")
# print(f"AT10: T60 varia de {min(at10):.2f}s a {max(at10):.2f}s")
# %%


############################################################################################################
# %%
# SNR com ventiladores


import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 0.8

# Dados de T60 médio para cada sala
frequencias = [125, 250, 500, 1000, 2000, 4000, 8000]
frequencias_labels = ['125', '250', '500', '1000', '2000', '4000', '8000']

# Dados das tabelas
at5 = [15.88, 15.78, 6.02, 1.40, 12.37, 20.84, 18.13]
at9 = [15.08, 10.91, 7.08, 10.92, 5.84, 11.55, 14.98]
at10 = [17.11, 9.76, 8.47, 2.64, 10.29, 19.93, 18.70]

# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar linhas com marcadores
ax.plot(frequencias, at5, marker='o', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT5', color='#1976D2')

ax.plot(frequencias, at9, marker='s', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9', color="#EC9B44")

# ax.plot(frequencias, at9_config2, marker='^', linestyle='-', linewidth=1.5, 
#         markersize=6, label='AT9 - Config. 2', color="#65BEEE")

ax.plot(frequencias, at10, marker='D', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT10', color='#D32F2F')

# Configurar eixos
ax.set_xlabel('Frequência [Hz]', fontsize=18, fontweight='normal')
ax.set_ylabel('SNR [dB]', fontsize=18, fontweight='normal')

# Configurar escala logarítmica no eixo x
ax.set_xscale('log')
ax.set_xticks(frequencias)
ax.set_xticklabels(frequencias_labels)

# Configurar limites do eixo y
ax.set_ylim(0.0, 40.0)
ax.set_yticks(np.arange(0, 42, 2.0))

# Grade
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legenda
ax.legend(loc='upper right', frameon=True, fontsize=15, 
          edgecolor='black', fancybox=False, shadow=False)

# Ajustar layout
plt.tight_layout()

# Salvar figura
plt.savefig('SNR_CV_comparacao_2_apre.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Exibir
plt.show()

# print("\nResumo dos dados:")
# print(f"AT5: T60 varia de {min(at5):.2f}s a {max(at5):.2f}s")
# print(f"AT9 Config.1: T60 varia de {min(at9_config1):.2f}s a {max(at9_config1):.2f}s")
# print(f"AT9 Config.2: T60 varia de {min(at9_config2):.2f}s a {max(at9_config2):.2f}s")
# print(f"AT10: T60 varia de {min(at10):.2f}s a {max(at10):.2f}s")
# %%

############################################################################################################
# %%
# U50


import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 0.8

# Dados de T60 médio para cada sala
frequencias = [125, 250, 500, 1000, 2000, 4000, 8000]
frequencias_labels = ['125', '250', '500', '1000', '2000', '4000', '8000']

# Dados das tabelas (sem ventiladores)
at5 = [-5.83, -1.10, -1.54, 0.33, 0.77, 0.98, 2.69]
at9_config1 = [-6.85, -7.49, -7.08, -5.65, -5.10, -4.12, -0.85]
at9_config2 = [-9.86, -7.12, -7.03, -5.24, -4.69, -3.61, -0.45]
at10 = [-3.11, -1.55, 1.57, 1.33, 2.99, 1.68, 3.48]

# # Dados das tabelas (com ventiladores)

# at5 = [-5.92, -1.25, -3.05, -3.60, 0.29, 0.90, 2.72]
# at9_config1 = [-6.99, -7.88, -8.00, -6.07, -6.38, -4.53, -1.07]
# at9_config2 = [-10.01, -7.52, -7.95, -5.68, -6.00, -4.02, -0.68]
# at10 = [-3.11, -2.32, 0.36, -2.31, 1.91, 1.57, 3.43]


# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar linhas com marcadores
ax.plot(frequencias, at5, marker='o', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT5', color='#1976D2')

ax.plot(frequencias, at9_config1, marker='s', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 1', color="#EC9B44")

ax.plot(frequencias, at9_config2, marker='^', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 2', color="#65BEEE")

ax.plot(frequencias, at10, marker='D', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT10', color='#D32F2F')

# Configurar eixos
ax.set_xlabel('Frequência [Hz]', fontsize=18, fontweight='normal')
ax.set_ylabel('U$_{50}$ [dB]', fontsize=18, fontweight='normal')

# Configurar escala logarítmica no eixo x
ax.set_xscale('log')
ax.set_xticks(frequencias)
ax.set_xticklabels(frequencias_labels)

# Configurar limites do eixo y
ax.set_ylim(-12.0, 7.0)
ax.set_yticks(np.arange(-12.0, 8.0, 1.0))

# Grade
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legenda
ax.legend(loc='upper left', frameon=True, fontsize=15, 
          edgecolor='black', fancybox=False, shadow=False)

# Ajustar layout
plt.tight_layout()

# Salvar figura
plt.savefig('U50_comparacao_apre.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Exibir
plt.show()

# print("\nResumo dos dados:")
# print(f"AT5: T60 varia de {min(at5):.2f}s a {max(at5):.2f}s")
# print(f"AT9 Config.1: T60 varia de {min(at9_config1):.2f}s a {max(at9_config1):.2f}s")
# print(f"AT9 Config.2: T60 varia de {min(at9_config2):.2f}s a {max(at9_config2):.2f}s")
# print(f"AT10: T60 varia de {min(at10):.2f}s a {max(at10):.2f}s")

# %%

#############################################################################3
# %%
# IF


import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 0.8

# Dados de T60 médio para cada sala
frequencias = [125, 250, 500, 1000, 2000, 4000, 8000]
frequencias_labels = ['125', '250', '500', '1000', '2000', '4000', '8000']

# # Dados das tabelas (sem ventiladores)
# at5 = [64.9498918, 97.27632, 95.8510192, 99.6676518, 99.7139398, 99.6216448, 96.11877820000001]
# at9_config1 = [53.063995000000006, 44.7158862, 50.142916799999995, 66.866395, 72.38592, 80.96421280000001, 97.941595]
# at9_config2 = [7.823755200000008, 49.625852800000004, 50.7854758, 71.0290512, 76.1706382, 84.7916302, 98.788155]
# at10 = [88.12081020000001, 95.814855, 98.96680380000001, 99.3035718, 94.9989262, 98.7801888, 92.8454448]

# Dados das tabelas (com ventiladores)
at5 = [63.971276800000005, 96.826875, 88.492155, 84.86232, 99.6473542, 99.66552, 96.0135808]
at9_config1 = [51.296506199999996, 39.292132800000005, 37.572, 62.3100838, 58.7574528, 77.5711758, 97.36168380000001]
at9_config2 = [5.172046200000011, 44.30772480000001, 38.291655, 66.55074880000001, 63.09, 90.00612480000001, 98.33414880000001]
at10 = [88.12081020000001, 92.5269088, 99.68111520000001, 92.57597820000001, 98.3244622, 98.96680380000001, 93.0836238]


# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar linhas com marcadores
ax.plot(frequencias, at5, marker='o', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT5', color='#1976D2')

ax.plot(frequencias, at9_config1, marker='s', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 1', color="#EC9B44")

ax.plot(frequencias, at9_config2, marker='^', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT9 - Config. 2', color="#65BEEE")

ax.plot(frequencias, at10, marker='D', linestyle='-', linewidth=1.5, 
        markersize=6, label='AT10', color='#D32F2F')

# Configurar eixos
ax.set_xlabel('Frequência [Hz]', fontsize=18, fontweight='normal')
ax.set_ylabel('IF [%]', fontsize=18, fontweight='normal')

# Configurar escala logarítmica no eixo x
ax.set_xscale('log')
ax.set_xticks(frequencias)
ax.set_xticklabels(frequencias_labels)

# Configurar limites do eixo y
ax.set_ylim(0.0, 100.0)
ax.set_yticks(np.arange(0.0, 105.0, 5.0))

# Grade
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legenda
ax.legend(loc='lower right', frameon=True, fontsize=15, 
          edgecolor='black', fancybox=False, shadow=False)

# Ajustar layout
plt.tight_layout()


# Salvar figura
plt.savefig('IF_comparacao_cv_apre.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Exibir
plt.show()

# print("\nResumo dos dados:")
# print(f"AT5: T60 varia de {min(at5):.2f}s a {max(at5):.2f}s")
# print(f"AT9 Config.1: T60 varia de {min(at9_config1):.2f}s a {max(at9_config1):.2f}s")
# print(f"AT9 Config.2: T60 varia de {min(at9_config2):.2f}s a {max(at9_config2):.2f}s")
# print(f"AT10: T60 varia de {min(at10):.2f}s a {max(at10):.2f}s")
# %%
