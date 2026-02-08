
#%%
import numpy as np
from scipy.signal import butter, sosfilt, convolve, filtfilt, hilbert
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Frequências centrais das bandas de oitava
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200  # Frequência de amostragem típica do sonômetro


# fs_lista = []
# audio = []
# audio_cortado = []
# t = []
# t_cortado = []
# max_pos_linear = np.zeros(8, dtype=int)
# max_pos = np.zeros(8, dtype=int)
# indice_inicio = np.zeros(8, dtype=int)
# indice_fim = np.zeros(8, dtype=int)
# t_inicio = np.zeros(8)
# t_fim = np.zeros(8)
# for i in range(1,9):
#     fs, audio_data = wav.read(f"AT10_explosao{i}.wav")
#     audio.append(audio_data)
#     fs_lista.append(fs)
#     audio[i-1] = audio[i-1] / np.max(np.abs(audio[i-1]))
#     t_data = np.arange(0, len(audio[i-1])/fs, 1/fs)
#     t.append(t_data)
#     max_pos_linear[i-1] = np.argmax(audio[i-1])
#     max_pos[i-1] = max_pos_linear[i-1]
#     soma_indice = 28230-1
#     indice_inicio[i-1] = max_pos[i-1] - 600
#     indice_fim[i-1] = max_pos[i-1] + soma_indice
#     t_inicio[i-1] = t[i-1][indice_inicio[i-1]]
#     t_fim[i-1] = t[i-1][indice_fim[i-1]-1]
#     audio_cortado_data = audio[i-1][indice_inicio[i-1]:indice_fim[i-1]]
#     audio_cortado.append(audio_cortado_data)
#     t_cortado_data = t[i-1][indice_inicio[i-1]:indice_fim[i-1]]
#     t_cortado.append(t_cortado_data)

#     # Plot 1: Curva de decaimento
#     plt.figure()
#     plt.plot(t_cortado[i-1], audio_cortado[i-1], color="dodgerblue")
#     plt.xlabel("Tempo [s]")
#     plt.ylabel("Nível normalizado [dB]")
#     # plt.title(f"Curva de decaimento")
#     plt.grid(True)
#     # plt.ylim(0, 1.0)
#     plt.tight_layout()
#     # plt.savefig(f"curva_decaimento.png")
#     plt.show()
    
# audio_cortado_final = np.array(audio_cortado)
# len(audio_cortado_final)
# t_cortado_final = np.array(t_cortado)
# fs = np.mean(fs_lista)

# audio_media = np.mean(audio_cortado_final, axis=0)
# t_media = np.mean(t_cortado_final, axis=0)

# # Plot 1: Curva de decaimento
# plt.figure()
# plt.plot(t_media, audio_media, color="dodgerblue")
# plt.xlabel("Tempo [s]")
# plt.ylabel("Nível normalizado [dB]")
# # plt.title(f"Curva de decaimento")
# plt.grid(True)
# # plt.ylim(0, 1.0)
# plt.tight_layout()
# # plt.savefig(f"c5urva_decaimento.png")
# plt.show()

# audio = audio_media
# t = t_media


# Carregamento do .wav
# fs, audio = wav.read("01dB_explosao.wav")
fs, audio = wav.read(f"AT10_explosao3.wav")
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


# --- Filtro de banda de oitava ---
def filtro_banda(data, fs, f_centro):
    f1 = f_centro / np.sqrt(2)
    f2 = f_centro * np.sqrt(2)
    sos = signal.butter(4, [f1, f2], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)



def rms(signal):
    return np.sqrt(np.mean(signal**2))

def filtro_passa_baixa(signal, fc_max, fs_signal, order=4):
    fc = min(fc_max, 80.0)  # Usar 50 Hz como frequência de corte
    nyquist = fs_signal / 2
    
    if fc >= nyquist:
        return signal
    
    try:
        sos = butter(order, fc / nyquist, btype='low', output='sos')
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal
    except Exception as e:
        print(f"ERRO no filtro: {e}")
        return signal

def calcular_intensidade_instantanea(signal, window_size=None):
    """
    Calcula a intensidade instantânea real do sinal usando envelope de Hilbert
    ou janela deslizante, em vez de assumir forma cossenoidal
    """
    if window_size is None:
        # Usar transformada de Hilbert para envelope
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        intensity = envelope**2
    else:
        # Método alternativo: janela deslizante para RMS
        intensity = np.zeros_like(signal)
        half_window = window_size // 2
        
        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window + 1)
            intensity[i] = np.mean(signal[start:end]**2)
    
    return intensity

def calcular_amplitude_modulacao_fft(intensity_envelope, fm, fs_signal):
    """
    Calcula amplitude de modulação usando FFT para ser mais preciso
    """
    N = len(intensity_envelope)
    
    # Remover componente DC
    intensity_ac = intensity_envelope - np.mean(intensity_envelope)
    dc_component = np.mean(intensity_envelope)
    
    if dc_component <= 0:
        return 0.0
    
    # Calcular FFT
    fft_result = np.fft.fft(intensity_ac)
    freqs = np.fft.fftfreq(N, 1/fs_signal)
    
    # Encontrar o bin de frequência mais próximo de fm
    fm_idx = np.argmin(np.abs(freqs - fm))
    
    # Amplitude na frequência de modulação
    amplitude_complex = fft_result[fm_idx]
    amplitude = 2 * np.abs(amplitude_complex) / N
    
    # Amplitude de modulação normalizada
    m = amplitude / dc_component
    
    return m

def calcular_amplitude_modulacao_correlacao(intensity_envelope, fm, fs_signal):
    """
    Método baseado em correlação com seno e cosseno (mais robusto)
    """
    N = len(intensity_envelope)
    t_vec = np.arange(N) / fs_signal
    
    # Remover DC
    mean_intensity = np.mean(intensity_envelope)
    if mean_intensity <= 0:
        return 0.0
    
    intensity_ac = intensity_envelope - mean_intensity
    
    # Correlação com cosseno e seno
    cos_ref = np.cos(2*np.pi*fm*t_vec)
    sin_ref = np.sin(2*np.pi*fm*t_vec)
    
    # Calcular correlações normalizadas
    cos_corr = 2 * np.mean(intensity_ac * cos_ref)
    sin_corr = 2 * np.mean(intensity_ac * sin_ref)
    
    # Amplitude de modulação
    m = np.sqrt(cos_corr**2 + sin_corr**2) / mean_intensity
    
    return m

def L_equivalente(audio, p_ref):
    # --- Cálculo de Leq global ---
    Leq = 10 * np.log10(np.mean((audio / p_ref)**2))
    return Leq

def L_max(audio, p_ref):
    # --- Cálculo de L max ---
    p_max = np.max(np.abs(audio))  # Valor máximo absoluto em Pa
    L_max = 20 * np.log10(p_max / p_ref)
    return L_max

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

# %%
