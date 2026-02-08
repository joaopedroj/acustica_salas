

# %%

# Calculo do SNR


import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import parametros
import claridade

f_central_lista = parametros.f_central_lista

p_ref = 20e-6                    # Pressão de referência para dB SPL (Pa)

# --- Leitura do arquivo .wav ---
fs, audio1 = wav.read("AT5_ruido rosa mic1.wav") # ruído rosa + fundo
fs, audio2 = wav.read("AT5_ruido fundo mic1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT5_ruido rosa ventilador2 mic1.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT5_ventilador2 mic1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT10_ruido rosa1.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT10_ruido fundo1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT10_ruido rosa ventilador.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT10_ruido ventilador1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT9_ruido rosa3.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT9_ruido fundo.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT9_ruido rosa ventilador2.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT9_ruido ventilador.wav") # ruído fundo

def UTR(C80, SNR):
    U80 = 10 * np.log10((10**(C80/10))/(1+(((10**(C80/10))+1)*(10**(-SNR/10)))))
    return U80

def audio_pascal(audio):
    # --- Parâmetros ---
    mic_sensitivity = 50e-3          # 50 mV/Pa
    full_scale_voltage = 1.0         # ±1 V = 2 Vpp
    max_duration = 121               # Duração máxima em segundos

    # --- Limitação do sinal a 120 segundos ---
    max_samples = int(fs * max_duration)  # Número máximo de amostras para 120s
    if len(audio) > max_samples:
        audio = audio[:max_samples]       # Corta o sinal em 120 segundos
        # print(f"Áudio cortado para {max_duration} segundos")
    # else:
    #     print(f"Áudio original tem {len(audio)/fs:.2f} segundos - não foi cortado")

    # --- Conversão do sinal digital para tensão ---
    if audio.dtype == np.int16:
        max_val = 2**15              # 32768 (16-bit)
    elif audio.dtype == np.int32:
        max_val = 2**31
    else:
        max_val = np.max(np.abs(audio))  # Para casos em float32

    # Normalização do sinal (valor entre -1 e 1)
    audio_norm = audio / max_val

    # --- Conversão da tensão para pressão sonora (Pa) ---
    # tensão = audio_norm * full_scale_voltage
    # pressão = tensão / sensibilidade
    audio_pa = (audio_norm * full_scale_voltage) / mic_sensitivity  # agora em Pascal

    return audio_pa


def L_equivalente(audio, p_ref):
    # --- Cálculo de Leq global ---
    Leq = 10 * np.log10(np.mean((audio / p_ref)**2))
    return Leq



# --- Conversão para dB SPL ---
audio_db_1 = 20 * np.log10(np.abs(audio_pascal(audio1)) / p_ref + 1e-12)  # evita log(0)
audio_db_2 = 20 * np.log10(np.abs(audio_pascal(audio2)) / p_ref + 1e-12)  # evita log(0)

# --- Tempo para eixo X ---
t1 = np.arange(0, len(audio_db_1)) / fs
t2 = np.arange(0, len(audio_db_2)) / fs



# --- Plotagem ---
# Plot audio 1 
plt.figure(figsize=(10, 4))
plt.plot(t1, audio_db_1, label="", lw=1.0)
plt.xlabel("Tempo [s]")
plt.ylabel("NPS [dB]")
#plt.title("NPS (dB)")
plt.text(0.02 * t1[-1], 1.15 * np.max(audio_db_1), f"Leq = {L_equivalente(audio_pascal(audio1), p_ref):.2f} dB", fontsize=12, color='black',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8)
         )
plt.grid(True)
plt.tight_layout()
# --- Marcações nos eixos ---
xticks_interval = 5  # segundos
xticks = np.arange(0, t1[-1], xticks_interval)
plt.xticks(xticks)
yticks_interval = 10  # segundos
yticks = np.arange(0, t1[-1], yticks_interval)
plt.yticks(yticks)
plt.xlim(0, 120)
plt.ylim(0, 100)
# plt.savefig(f"ruido_de_fundo_AT9_1.png")
plt.show()

# Plot audio 2
plt.figure(figsize=(10, 4))
plt.plot(t2, audio_db_2, label="", lw=1.0)
plt.xlabel("Tempo [s]")
plt.ylabel("NPS [dB]")
#plt.title("NPS (dB)")
plt.text(0.02 * t2[-1], 1.15 * np.max(audio_db_2), f"Leq = {L_equivalente(audio_pascal(audio2), p_ref):.2f} dB", fontsize=12, color='black',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8)
         )
plt.grid(True)
plt.tight_layout()
# --- Marcações nos eixos ---
xticks_interval = 5  # segundos
xticks = np.arange(0, t2[-1], xticks_interval)
plt.xticks(xticks)
yticks_interval = 10  # segundos
yticks = np.arange(0, t2[-1], yticks_interval)
plt.yticks(yticks)
plt.xlim(0, 120)
plt.ylim(0, 100)
# plt.savefig(f"ruido_de_fundo_AT9_2.png")
plt.show()

SNR = []
for f_central in f_central_lista:
    # Filtragem
    filtered_1 = parametros.filtro_banda(audio_pascal(audio1), fs, f_central)
    filtered_2 = parametros.filtro_banda(audio_pascal(audio2), fs, f_central)

    Ln = L_equivalente(filtered_2, p_ref) # Leq do ruído de fundo
    # print(f"Leq do ruído de fundo ({f_central}): {Ln}")
    Lsn = L_equivalente(filtered_1, p_ref) # Leq do ruído rosa + ruído de fundo
    # print(f"Leq do ruído rosa + ruído de fundo ({f_central}): {Lsn}")
    Ls = 10*np.log10((10**(Lsn/10)) - (10**(Ln/10))) # Leq do ruído rosa
    # print(f"Leq do ruído rosa ({f_central}): {Ls}")

    #Calculo do SNR
    snr = Ls - Ln
    SNR.append(snr)
    print(f"Banda {f_central} Hz: SNR ≈ {snr:.3f}")

    # # Cálculo e plotagem do audio filtrado e o Leq filtrado
    # audio_db_1_filtered = 20 * np.log10(np.abs(audio_pascal(filtered_1)) / p_ref + 1e-12)
    # audio_db_2_filtered = 20 * np.log10(np.abs(audio_pascal(filtered_2)) / p_ref + 1e-12)
    # t1_f = np.arange(0, len(audio_db_1_filtered)) / fs
    # t2_f = np.arange(0, len(audio_db_2_filtered)) / fs

    # # Plot audio 1 
    # plt.figure(figsize=(10, 4))
    # plt.plot(t1_f, audio_db_1_filtered, label="")
    # plt.xlabel("Tempo (s)")
    # plt.ylabel("NPS (dB)")
    # plt.title(f"{f_central} Hz ")
    # plt.text(0.02 * t1_f[-1], 1.15 * np.max(audio_db_1_filtered), f"Leq = {L_equivalente(audio_pascal(filtered_1), p_ref):.2f} dB", fontsize=12, color='black',
    #         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8)
    #         )
    # plt.grid(True)
    # plt.tight_layout()
    # # --- Marcações nos eixos ---
    # xticks_interval = 5  # segundos
    # xticks = np.arange(0, t1_f[-1], xticks_interval)
    # plt.xticks(xticks)
    # yticks_interval = 10  # segundos
    # yticks = np.arange(0, t1_f[-1], yticks_interval)
    # plt.yticks(yticks)
    # plt.xlim(0, np.max(t1_f))
    # plt.ylim(0, 100)
    # # plt.savefig(f"rosa2_{L_equivalente(audio1)}dB.png")
    # plt.show()

    # # Plot audio 2
    # plt.figure(figsize=(10, 4))
    # plt.plot(t2_f, audio_db_2_filtered, label="")
    # plt.xlabel("Tempo (s)")
    # plt.ylabel("NPS (dB)")
    # plt.title(f"{f_central} Hz ")
    # plt.text(0.02 * t2_f[-1], 1.15 * np.max(audio_db_2_filtered), f"Leq = {L_equivalente(audio_pascal(filtered_2), p_ref):.2f} dB", fontsize=12, color='black',
    #         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8)
    #         )
    # plt.grid(True)
    # plt.tight_layout()
    # # --- Marcações nos eixos ---
    # xticks_interval = 5  # segundos
    # xticks = np.arange(0, t2_f[-1], xticks_interval)
    # plt.xticks(xticks)
    # yticks_interval = 10  # segundos
    # yticks = np.arange(0, t2_f[-1], yticks_interval)
    # plt.yticks(yticks)
    # plt.xlim(0, np.max(t2_f))
    # plt.ylim(0, 100)
    # # plt.savefig(f"rosa2_{L_equivalente(audio1)}dB.png")
    # plt.show()

# C80 = claridade.c_50
# U80 = []

# for i in range(0, len(f_central_lista)):
#     u = UTR(C80[i], SNR[i])
#     U80.append(u)
#     print(f"Banda {f_central_lista[i]} Hz: U80 ≈ {u:.3f}")

# %%
##########################################################################################################################
# %%
# Calculo do U50

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def UTR(C80, SNR):
    U80 = 10 * np.log10((10**(C80/10))/(1+(((10**(C80/10))+1)*(10**(-SNR/10)))))
    return U80

def audio_pascal(audio):
    # --- Parâmetros ---
    mic_sensitivity = 50e-3          # 50 mV/Pa
    full_scale_voltage = 1.0         # ±1 V = 2 Vpp
    max_duration = 121               # Duração máxima em segundos

    # --- Limitação do sinal a 120 segundos ---
    max_samples = int(fs * max_duration)  # Número máximo de amostras para 120s
    if len(audio) > max_samples:
        audio = audio[:max_samples]       # Corta o sinal em 120 segundos
        # print(f"Áudio cortado para {max_duration} segundos")
    else:
        print(f"Áudio original tem {len(audio)/fs:.2f} segundos - não foi cortado")

    # --- Conversão do sinal digital para tensão ---
    if audio.dtype == np.int16:
        max_val = 2**15              # 32768 (16-bit)
    elif audio.dtype == np.int32:
        max_val = 2**31
    else:
        max_val = np.max(np.abs(audio))  # Para casos em float32

    # Normalização do sinal (valor entre -1 e 1)
    audio_norm = audio / max_val

    # --- Conversão da tensão para pressão sonora (Pa) ---
    # tensão = audio_norm * full_scale_voltage
    # pressão = tensão / sensibilidade
    audio_pa = (audio_norm * full_scale_voltage) / mic_sensitivity  # agora em Pascal

    return audio_pa


def L_equivalente(audio, p_ref):
    # --- Cálculo de Leq global ---
    Leq = 10 * np.log10(np.mean((audio / p_ref)**2))
    return Leq

import parametros

f_central_lista = parametros.f_central_lista

p_ref = 20e-6                    # Pressão de referência para dB SPL (Pa)

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT5_ruido rosa mic1.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT5_ruido fundo mic1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT5_ruido rosa ventilador2 mic1.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT5_ventilador2 mic1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT10_ruido rosa1.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT10_ruido fundo1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT10_ruido rosa ventilador.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT10_ruido ventilador1.wav") # ruído fundo

# # --- Leitura do arquivo .wav ---
# fs, audio1 = wav.read("AT9_ruido rosa3.wav") # ruído rosa + fundo
# fs, audio2 = wav.read("AT9_ruido fundo.wav") # ruído fundo

# --- Leitura do arquivo .wav ---
fs, audio1 = wav.read("AT9_ruido rosa ventilador2.wav") # ruído rosa + fundo
fs, audio2 = wav.read("AT9_ruido ventilador.wav") # ruído fundo


# --- Conversão para dB SPL ---
audio_db_1 = 20 * np.log10(np.abs(audio_pascal(audio1)) / p_ref + 1e-12)  # evita log(0)
audio_db_2 = 20 * np.log10(np.abs(audio_pascal(audio2)) / p_ref + 1e-12)  # evita log(0)

# --- Tempo para eixo X ---
t1 = np.arange(0, len(audio_db_1)) / fs
t2 = np.arange(0, len(audio_db_2)) / fs


SNR = []
for f_central in f_central_lista:
    # Filtragem
    filtered_1 = parametros.filtro_banda(audio_pascal(audio1), fs, f_central)
    filtered_2 = parametros.filtro_banda(audio_pascal(audio2), fs, f_central)

    Ln = L_equivalente(filtered_2, p_ref) # Leq do ruído de fundo
    Lsn = L_equivalente(filtered_1, p_ref) # Leq do ruído rosa + ruído de fundo
    Ls = 10*np.log10((10**(Lsn/10)) - (10**(Ln/10))) # Leq do ruído rosa

    #Calculo do SNR
    snr = Ls - Ln
    SNR.append(snr)
    # print(f"Banda {f_central} Hz: SNR ≈ {snr:.3f}")


Lista_ATs = [9]

for j in Lista_ATs:
    for at in range(1,17):
        # Carregamento do .wav
        fs, audio = wav.read(f"AT{j}_explosão{at}.wav")
        audio = audio / np.max(np.abs(audio))  # Normalização
        t = np.arange(0, (len(audio)-1)/fs + 1/fs, 1/fs)

        # --- Definição da janela ---
        max_pos_linear = np.argmax(audio)
        max_pos = np.unravel_index(max_pos_linear, audio.shape)
        soma_indice = 28230-1
        indice_inicio = max_pos[0] - 600
        indice_fim = max_pos[0] + soma_indice
        t_inicio = t[indice_inicio]
        t_fim = t[indice_fim]
        audio_cortado = audio[indice_inicio:indice_fim]
        t_cortado = np.arange(t_inicio, t_fim, 1/fs)

        c_50 = []
        d_50 = []
        # --- Loop para cada banda de oitava ---
        for f_central in f_central_lista:
            # Filtragem
            filtered = parametros.filtro_banda(audio_cortado, fs, f_central)

            # Curva de Schroeder
            energy = filtered**2
            schroeder = np.cumsum(energy[::-1])[::-1]
            schroeder /= np.max(schroeder)
            schroeder_db = 10 * np.log10(schroeder)
            t_sch = np.arange(len(schroeder_db)) / fs


            limite_50ms = int(0.05 * fs)  # 50 milissegundos em amostras
            # Cálculo de D50
            energia_50ms = np.trapz(energy[:limite_50ms], dx=1/fs)
            energia_total = np.trapz(energy, dx=1/fs)
            D50 = energia_50ms / energia_total
            C50 = 10*np.log10(D50)

            # Cálculo de C50
            energia_pre50 = np.trapz(energy[:limite_50ms], dx=1/fs)
            energia_pos50 = np.trapz(energy[limite_50ms:], dx=1/fs)
            C50 = 10 * np.log10(energia_pre50 / energia_pos50)
            c_50.append(C50)
            d_50.append(D50)
            # print(f"AT{j} impulso {at} - Banda {f_central} Hz: C50 ≈ {C50:.3f}")
            U50 = []

        for i in range(0, len(f_central_lista)):
            u = UTR(c_50[i], SNR[i])
            # print(f"AT{j} impulso {at} - Banda {f_central_lista[i]} Hz: SNR ≈ {SNR[i]:.3f}")
            # print(f"AT{j} impulso {at} - Banda {f_central_lista[i]} Hz: C50 ≈ {c_50[i]:.3f}")
            U50.append(u)
            print(f"AT{j} impulso {at} - Banda {f_central_lista[i]} Hz: U50 ≈ {u:.3f}")

# %%
