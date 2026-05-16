import os
import sys
import glob
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Permite execução standalone — adiciona scripts/ ao sys.path para achar parametros.py
_PARENT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import parametros

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'audio')

f_central_lista = parametros.f_central_lista

p_ref = 20e-6


def UTR(C80, SNR):
    U80 = 10 * np.log10((10**(C80/10)) / (1 + (((10**(C80/10)) + 1) * (10**(-SNR/10)))))
    return U80


def audio_pascal(audio, fs):
    mic_sensitivity = 50e-3
    full_scale_voltage = 1.0
    max_duration = 121

    max_samples = int(fs * max_duration)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    if audio.dtype == np.int16:
        max_val = 2**15
    elif audio.dtype == np.int32:
        max_val = 2**31
    else:
        max_val = np.max(np.abs(audio))

    audio_norm = audio / max_val
    audio_pa = (audio_norm * full_scale_voltage) / mic_sensitivity
    return audio_pa


def L_equivalente(audio, p_ref):
    return 10 * np.log10(np.mean((audio / p_ref)**2))


def calcular_snr(pink_file, background_file, f_central_lista, p_ref):
    fs, audio1 = wav.read(pink_file)
    _, audio2 = wav.read(background_file)

    SNR = []
    for f_central in f_central_lista:
        filtered_1 = parametros.filtro_banda(audio_pascal(audio1, fs), fs, f_central)
        filtered_2 = parametros.filtro_banda(audio_pascal(audio2, fs), fs, f_central)

        Ln = L_equivalente(filtered_2, p_ref)
        Lsn = L_equivalente(filtered_1, p_ref)
        Ls = 10 * np.log10((10**(Lsn/10)) - (10**(Ln/10)))
        snr = Ls - Ln
        SNR.append(snr)
        print(f"Banda {f_central} Hz: SNR ≈ {snr:.3f}")

    return SNR, fs, audio1, audio2


def plotar_nps(audio, fs, p_ref):
    audio_db = 20 * np.log10(np.abs(audio_pascal(audio, fs)) / p_ref + 1e-12)
    t = np.arange(0, len(audio_db)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, audio_db, lw=1.0)
    plt.xlabel("Tempo [s]")
    plt.ylabel("NPS [dB]")
    plt.text(0.02 * t[-1], 1.15 * np.max(audio_db),
             f"Leq = {L_equivalente(audio_pascal(audio, fs), p_ref):.2f} dB",
             fontsize=12, color='black',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8))
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(np.arange(0, t[-1], 5))
    plt.yticks(np.arange(0, t[-1], 10))
    plt.xlim(0, 120)
    plt.ylim(0, 100)
    plt.show()


if __name__ == "__main__":
    SNR_at5, fs_at5, audio1_at5, audio2_at5 = calcular_snr(
        os.path.join(AUDIO_DIR, 'AT5_noise_pink_mic1.wav'),
        os.path.join(AUDIO_DIR, 'AT5_noise_background_mic1.wav'),
        f_central_lista, p_ref
    )

    plotar_nps(audio1_at5, fs_at5, p_ref)
    plotar_nps(audio2_at5, fs_at5, p_ref)

    SNR_at9, _, _, _ = calcular_snr(
        os.path.join(AUDIO_DIR, 'AT9_noise_pink_ventilador2.wav'),
        os.path.join(AUDIO_DIR, 'AT9_noise_ventilator_1.wav'),
        f_central_lista, p_ref
    )

    at9_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT9_impulse_[0-9]*.wav')))

    for filepath in at9_files:
        fs, audio = wav.read(filepath)
        audio = audio / np.max(np.abs(audio))
        t = np.arange(0, (len(audio)-1)/fs + 1/fs, 1/fs)

        max_pos = np.unravel_index(np.argmax(audio), audio.shape)
        soma_indice = 28230 - 1
        indice_inicio = max_pos[0] - 600
        indice_fim = max_pos[0] + soma_indice
        audio_cortado = audio[indice_inicio:indice_fim]

        fname = os.path.splitext(os.path.basename(filepath))[0]

        c_50 = []
        d_50 = []

        for f_central in f_central_lista:
            filtered = parametros.filtro_banda(audio_cortado, fs, f_central)
            energy = filtered**2

            limite_50ms = int(0.05 * fs)
            energia_pre50 = np.trapz(energy[:limite_50ms], dx=1/fs)
            energia_pos50 = np.trapz(energy[limite_50ms:], dx=1/fs)
            energia_total = np.trapz(energy, dx=1/fs)
            D50 = energia_pre50 / energia_total
            C50 = 10 * np.log10(energia_pre50 / energia_pos50)
            c_50.append(C50)
            d_50.append(D50)

        U50 = []
        for i in range(len(f_central_lista)):
            u = UTR(c_50[i], SNR_at9[i])
            U50.append(u)
            print(f"AT9 impulso {fname} - Banda {f_central_lista[i]} Hz: U50 ≈ {u:.3f}")
