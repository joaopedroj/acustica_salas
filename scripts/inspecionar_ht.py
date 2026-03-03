import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio')


def processar_arquivo(audio, fs):
    max_pos = np.unravel_index(np.argmax(audio), audio.shape)
    soma_indice = 4 * fs - 1
    indice_inicio = max_pos[0] - 2000
    audio_cortado = audio[indice_inicio : max_pos[0] + soma_indice]
    t_cortado = np.arange(len(audio_cortado)) / fs

    plt.figure()
    plt.plot(t_cortado, audio_cortado, color="dodgerblue")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.grid(True)
    plt.xlim(0, 4)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()

    energy = audio_cortado**2
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder /= np.max(schroeder)
    schroeder_db = 10 * np.log10(schroeder + 1e-12)
    t_sch = t_cortado

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


if __name__ == "__main__":
    processar_arquivo(parametros.audio, parametros.fs)

    at9_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT9_impulse_[0-9]*.wav')))

    for filepath in at9_files:
        fs, audio = wav.read(filepath)
        audio = audio / np.max(np.abs(audio))
        processar_arquivo(audio, fs)
