# Todas as frequencias de banda de oitava (Schroeder + ajuste)
# %%
import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt

import parametros
audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t

# --- Definição da janela ---
max_pos_linear = np.argmax(audio)
max_pos = np.unravel_index(max_pos_linear, audio.shape)
soma_indice = int(1*fs) - 1
indice_inicio = max_pos[0]
indice_fim = max_pos[0] + soma_indice
t_inicio = t[indice_inicio]
t_fim = t[indice_fim]
audio_cortado = audio[indice_inicio:indice_fim]

# Criação do vetor de tempo baseado no número real de amostras
n_amostras = len(audio_cortado)
t_cortado = np.arange(n_amostras) / fs  # Tempo relativo começando em 0

# Plot 1: Curva de decaimento
plt.figure()
plt.plot(t_cortado, -audio_cortado, color="dodgerblue")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude [Pa]")
# plt.title(f"Curva de decaimento")
plt.grid(True)
plt.xlim(0, 0.5)
plt.ylim(-1, 1)
plt.xticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
plt.tight_layout()
plt.savefig(f"primeiras_reflexoes.png")
plt.show()

# --- Loop para cada banda de oitava ---
for f_central in f_central_lista:
    # Filtragem
    filtered = parametros.filtro_banda(audio_cortado, fs, f_central)

    # Curva de Schroeder (Resposta ao Impulso)
    energy = filtered**2
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder /= np.max(schroeder)
    schroeder_db = 10 * np.log10(schroeder + 1e-12)  # Evita log(0)
    # Usa o MESMO vetor de tempo
    t_sch = t_cortado  # Garante correspondência perfeita

    # Plot 2: Curva de Schroeder
    plt.figure()
    plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Nível [dB]")
    plt.title(f"{f_central} Hz")
    plt.grid(True)
    plt.xlim(0, 2.6)
    plt.ylim(-100, 10)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"schroeder_{int(f_central)}Hz.png")
    plt.show()

    # Plot 3: Energia instantânea + Schroeder
    energy_db = 10 * np.log10(energy / np.max(energy) + 1e-12)
    plt.figure()
    # plt.plot(t_sch, energy_db, color="dodgerblue")
    plt.plot(t_sch, energy_db, alpha=0.5, color="dodgerblue")
    plt.plot(t_sch, schroeder_db, linewidth=2, color="black")
    # plt.plot(t_sch, energy_db, label="Energia instantânea", alpha=0.5, color="dodgerblue")
    # plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", linewidth=2, color="orange")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Nível [dB]")
    plt.xlim(0, 1.2)
    plt.ylim(-100, 10)
    plt.grid(True)
    # plt.title(f"{f_central} Hz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"h(t)_sc_diag_{int(f_central)}Hz.png")
    plt.show()

    # # Plot teste
    # energy_db = 10 * np.log10(energy / np.max(energy) + 1e-12)
    # plt.figure()
    # plt.plot(t_sch, filtered, color="dodgerblue")
    # # plt.plot(t_sch, energy_db, label="Energia instantânea", alpha=0.5, color="dodgerblue")
    # # plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", linewidth=2, color="orange")
    # plt.xlabel("Tempo [s]")
    # plt.ylabel("Nível [dB]")
    # # plt.xlim(0, 1.0)
    # # plt.ylim(-100, 10)
    # plt.grid(True)
    # # plt.title(f"{f_central} Hz")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"RIS_{int(f_central)}Hz.png")
    # plt.show()

    # Plot 4: Ajuste linear para T20
    mask = (schroeder_db <= 0) & (schroeder_db >= -25)
    t_fit = t_sch[mask]
    y_fit = schroeder_db[mask]
    
    if len(t_fit) > 1:
        coeffs = np.polyfit(t_fit, y_fit, 1)
        T20 = -60 / coeffs[0]
        line_fit = np.polyval(coeffs, t_fit)
        line_full = np.polyval(coeffs, t_sch)

        plt.figure()
        plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
        plt.plot(t_sch, line_full, 'k--', label="Ajuste linear", linewidth=2)
        plt.xlabel("Tempo [s]")
        plt.ylabel("Nível [dB]")
        plt.title(f"{f_central} Hz")
        plt.grid(True)
        plt.xlim(0, 2.6)
        plt.ylim(-100, 10)
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")
        plt.show()

        print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
    else:
        print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")

# %%

# %%
##########################################################################################################
# Automação para todos os arquivos

import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

# Frequências centrais das bandas de oitava
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200  # Frequência de amostragem típica do sonômetro

for i in range(9,17):

    fs, audio = wav.read(f"AT9_explosão{i}.wav")
    # fs, audio = wav.read(f"AT9_explosão{i}.wav")
    audio = audio / np.max(np.abs(audio))  # Normalização
    t = np.arange(0, (len(audio)-1)/fs + 1/fs, 1/fs)

    # --- Definição da janela ---
    max_pos_linear = np.argmax(audio)
    max_pos = np.unravel_index(max_pos_linear, audio.shape)
    # soma_indice = int(1.0*fs) - 1
    # soma_indice = int(1.2*fs) - 1
    soma_indice = int(2.3*fs) - 1
    # soma_indice = int(3*fs) - 1
    indice_inicio = max_pos[0] - 2000
    # indice_inicio = max_pos[0] - 1000
    indice_fim = max_pos[0] + soma_indice
    t_inicio = t[indice_inicio]
    t_fim = t[indice_fim]
    audio_cortado = audio[indice_inicio:indice_fim]

    # Criação do vetor de tempo baseado no número real de amostras
    n_amostras = len(audio_cortado)
    t_cortado = np.arange(n_amostras) / fs  # Tempo relativo começando em 0

    # # Plot 1: Curva de decaimento
    # plt.figure()
    # plt.plot(t_cortado, audio_cortado, color="dodgerblue")
    # plt.xlabel("Tempo [s]")
    # plt.ylabel("Amplitude [Pa]")
    # # plt.title(f"Curva de decaimento")
    # plt.grid(True)
    # plt.xlim(0, 3)
    # plt.ylim(-1, 1)
    # plt.tight_layout()
    # # plt.savefig(f"curva_decaimento_AT9_{i}.png")
    # plt.show()

    # Curva de Resposta ao Impulso sem filtro de frequências
    energy_1 = audio_cortado**2
    t_en = t_cortado  # Garante correspondência perfeita

    # energy_db_1 = 10 * np.log10(energy_1 / np.max(energy_1))
    # plt.figure()
    # plt.plot(t_en, energy_db_1, color="dodgerblue")
    # plt.xlabel("Tempo [s]")
    # plt.ylabel("NPS [dB]")
    # plt.xlim(0, 1.0)
    # # plt.xlim(0, 3.0)
    # plt.ylim(-100, 7)
    # plt.grid(True)
    # # plt.title(f"{f_central} Hz")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"h(t)_schroeder_sem_filtro_AT10_{i}.png")
    # plt.show()


    # --- Loop para cada banda de oitava ---
    for f_central in f_central_lista:
        # Filtragem
        filtered = parametros.filtro_banda(audio_cortado, fs, f_central)
        

        # Curva de Schroeder (Resposta ao Impulso)
        energy = filtered**2
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder /= np.max(schroeder)
        schroeder_db = 10 * np.log10(schroeder + 1e-12)  # Evita log(0)
        # Usa o MESMO vetor de tempo
        t_sch = t_cortado  # Garante correspondência perfeita

        # # Plot 2: Curva de Schroeder
        # plt.figure()
        # plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
        # plt.xlabel("Tempo [s]")
        # plt.ylabel("Nível [dB]")
        # plt.title(f"{f_central} Hz")
        # plt.grid(True)
        # plt.xlim(0, 2.8)
        # plt.ylim(-100, 10)
        # plt.legend()
        # plt.tight_layout()
        # # plt.savefig(f"schroeder_{int(f_central)}Hz_AT10_{i}.png")
        # plt.show()

        # Plot 3: Energia instantânea + Schroeder
        energy_db = 10 * np.log10(energy / np.max(energy) + 1e-12)
        plt.figure()
        plt.plot(t_sch, energy_db, label="Energia instantânea", alpha=0.5, color="dodgerblue")
        plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", linewidth=2, color="orange")
        plt.xlabel("Tempo [s]")
        plt.ylabel("NPS [dB]")
        # plt.xlim(0, 1.2)
        plt.xlim(0, 2.8)
        plt.ylim(-120, 10)
        plt.grid(True)
        # plt.title(f"{f_central} Hz")
        plt.title(f"AT9 - Config. 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"h(t)_schroeder_{int(f_central)}Hz_AT9_{i}.png")
        plt.show()

        # # Plot 4: Ajuste linear para T20
        # mask = (schroeder_db <= 0) & (schroeder_db >= -25)
        # t_fit = t_sch[mask]
        # y_fit = schroeder_db[mask]
        
        # if len(t_fit) > 1:
        #     coeffs = np.polyfit(t_fit, y_fit, 1)
        #     T20 = -60 / coeffs[0]
        #     line_fit = np.polyval(coeffs, t_fit)
        #     line_full = np.polyval(coeffs, t_sch)

        #     plt.figure()
        #     plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
        #     plt.plot(t_sch, line_full, 'k--', label="Ajuste linear", linewidth=2)
        #     plt.xlabel("Tempo [s]")
        #     plt.ylabel("Nível [dB]")
        #     plt.title(f"{f_central} Hz")
        #     plt.grid(True)
        #     plt.xlim(0, 2.8)
        #     plt.ylim(-100, 10)
        #     plt.legend()
        #     plt.tight_layout()
        #     # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz_AT10_{i}.png")
        #     plt.show()

        #     print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
        # else:
        #     print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")

# %%

