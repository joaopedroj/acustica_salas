# %%

import numpy as np
from scipy.signal import butter, sosfilt, convolve, filtfilt, hilbert
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Frequências centrais das bandas de oitava
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200  # Frequência de amostragem típica do sonômetro

for i in range(1,17):

    # Carregamento do .wav
    # fs, audio = wav.read("01dB_explosao.wav")
    fs, audio = wav.read(f"AT9_explosão{i}.wav")
    audio = audio / np.max(np.abs(audio))  # Normalização
    t = np.arange(0, (len(audio)-1)/fs + 1/fs, 1/fs)


    # --- Definição da janela ---
    max_pos_linear = np.argmax(audio)
    max_pos = np.unravel_index(max_pos_linear, audio.shape)
    soma_indice = 4*fs - 1
    indice_inicio = max_pos[0] - 2000
    indice_fim = max_pos[0] + soma_indice
    t_inicio = t[indice_inicio]
    t_fim = t[indice_fim]
    audio_cortado = audio[indice_inicio:indice_fim]

    # Criação do vetor de tempo baseado no número real de amostras
    n_amostras = len(audio_cortado)
    t_cortado = np.arange(n_amostras) / fs  # Tempo relativo começando em 0

    # Plot 1: Curva de decaimento
    plt.figure()
    plt.plot(t_cortado, audio_cortado, color="dodgerblue")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude [Pa]")
    # plt.title(f"Curva de decaimento")
    plt.grid(True)
    plt.xlim(0, 4)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()


    
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
soma_indice = 2*fs- 1
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
t_inicio = t[indice_inicio]
t_fim = t[indice_fim]
audio_cortado = audio[indice_inicio:indice_fim]
# t_cortado = np.arange(t_inicio, t_fim, 1/fs)
t_cortado = np.arange(t_inicio - t_inicio, t_fim - t_inicio, 1/fs)

# Plot 1: Curva de decaimento
plt.figure()
# plt.plot(t_cortado[:-1], audio_cortado, color="dodgerblue")
plt.plot(t_cortado[::], audio_cortado, color="dodgerblue")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude [Pa]")
plt.title(f"Curva de decaimento")
plt.grid(True)
# plt.xlim(t[indice_inicio], t[indice_fim])
plt.xlim(t_cortado[0], t_cortado[-500])
plt.ylim(-1, 1)
plt.tight_layout()
# plt.savefig(f"curva_decaimento_AT9_16.png")
plt.show()



# Curva de Schroeder
energy = audio_cortado**2
schroeder = np.cumsum(energy[::-1])[::-1]
schroeder /= np.max(schroeder)
schroeder_db = 10 * np.log10(schroeder)
t_sch = np.arange(len(schroeder_db)) / fs


# Plot 3: Energia instantânea + Schroeder
energy_db = 10 * np.log10(energy / np.max(energy))
plt.figure()
plt.plot(t_sch, energy_db, label="Energia instantânea", color="dodgerblue")
plt.xlabel("Tempo [s]")
plt.ylabel("Nível [dB]")
plt.xlim(0, 4.8)
plt.ylim(-120, 10)
plt.grid(True)
# plt.title(f"{f_central} Hz")
plt.legend()
plt.tight_layout()
# plt.savefig(f"h(t)_schroeder_{int(f_central)}Hz.png")
plt.show()

print(len(audio_cortado))
print(len(t_cortado[:-1]))
print(len(energy))
print(len(energy_db))
print(len(schroeder))
print(len(schroeder_db))
print(len(t_sch))
print(energy_db)
print(t_sch)


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
soma_indice = 4*fs - 1
indice_inicio = max_pos[0] - 2000
indice_fim = max_pos[0] + soma_indice
t_inicio = t[indice_inicio]
t_fim = t[indice_fim]
audio_cortado = audio[indice_inicio:indice_fim]

# Criação do vetor de tempo baseado no número real de amostras
n_amostras = len(audio_cortado)
t_cortado = np.arange(n_amostras) / fs  # Tempo relativo começando em 0

# Plot 1: Curva de decaimento
plt.figure()
plt.plot(t_cortado, audio_cortado, color="dodgerblue")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude [Pa]")
# plt.title(f"Curva de decaimento")
plt.grid(True)
plt.xlim(0, 4)
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()

# Curva de Schroeder (Resposta ao Impulso)
energy = audio_cortado**2
schroeder = np.cumsum(energy[::-1])[::-1]
schroeder /= np.max(schroeder)
schroeder_db = 10 * np.log10(schroeder + 1e-12)  # Evita log(0)

# Usa o MESMO vetor de tempo
t_sch = t_cortado  # Garante correspondência perfeita

# Plot 2: Energia instantânea
energy_db = 10 * np.log10(energy / np.max(energy) + 1e-12)
plt.figure()
plt.plot(t_sch, energy_db, color="dodgerblue")
plt.xlabel("Tempo [s]")
plt.ylabel("Nível [dB]")
plt.xlim(0, 4)
plt.ylim(-120, 10)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Duração do áudio cortado: {len(audio_cortado)/fs:.3f} s")
print(f"Número de amostras: {len(audio_cortado)}")
print(f"Duração de t_cortado: {t_cortado[-1]:.3f} s")
print(f"Duração de t_sch: {t_sch[-1]:.3f} s")
# %%
