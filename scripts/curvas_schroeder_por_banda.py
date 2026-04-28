"""curvas_schroeder_por_banda.py — Visualização das curvas de Schroeder por banda."""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio')
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]


def computar_schroeder_normalizado(audio_cortado, fs, f_central):
    filtered = parametros.filtro_banda(audio_cortado, fs, f_central)
    energy = filtered ** 2
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder /= np.max(schroeder)
    schroeder_db = 10 * np.log10(schroeder + 1e-12)
    energy_db = 10 * np.log10(energy / np.max(energy) + 1e-12)
    return energy_db, schroeder_db


if __name__ == "__main__":
    _default = parametros.load_default_audio()
    audio_ref = _default['audio']
    fs = _default['fs']

    max_pos = int(np.argmax(audio_ref))
    soma_indice = int(1 * fs) - 1
    audio_cortado_ref = audio_ref[max_pos: max_pos + soma_indice]
    t_cortado_ref = np.arange(len(audio_cortado_ref)) / fs

    plt.figure()
    plt.plot(t_cortado_ref, -audio_cortado_ref, color="dodgerblue")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.grid(True)
    plt.xlim(0, 0.5)
    plt.ylim(-1, 1)
    plt.xticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    plt.tight_layout()
    plt.show()

    for f_central in f_central_lista:
        energy_db, schroeder_db = computar_schroeder_normalizado(
            audio_cortado_ref, fs, f_central)
        t_sch = t_cortado_ref

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
        plt.show()

        plt.figure()
        plt.plot(t_sch, energy_db, alpha=0.5, color="dodgerblue")
        plt.plot(t_sch, schroeder_db, linewidth=2, color="black")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Nível [dB]")
        plt.xlim(0, 1.2)
        plt.ylim(-100, 10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        resultado = parametros.ajustar_t60(t_sch, schroeder_db)
        if resultado is None:
            print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
            continue
        coeffs, _t_fit, T20, T30, T60 = resultado
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
        plt.show()

        print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s | "
              f"T30 ≈ {T30:.2f} s | T60 ≈ {T60:.2f} s")

    at9_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT9_impulse_[0-9]*.wav')))

    for filepath in at9_files:
        fs, audio = wav.read(filepath)
        audio = audio / np.max(np.abs(audio))

        max_pos = int(np.argmax(audio))
        soma_indice = int(2.3 * fs) - 1
        indice_inicio = max_pos - 2000
        audio_cortado = audio[indice_inicio: max_pos + soma_indice]
        t_cortado = np.arange(len(audio_cortado)) / fs

        fname = os.path.splitext(os.path.basename(filepath))[0]

        for f_central in f_central_lista:
            energy_db, schroeder_db = computar_schroeder_normalizado(
                audio_cortado, fs, f_central)
            t_sch = t_cortado

            plt.figure()
            plt.plot(t_sch, energy_db, label="Energia instantânea",
                     alpha=0.5, color="dodgerblue")
            plt.plot(t_sch, schroeder_db, label="Curva de Schroeder",
                     linewidth=2, color="orange")
            plt.xlabel("Tempo [s]")
            plt.ylabel("NPS [dB]")
            plt.xlim(0, 2.8)
            plt.ylim(-120, 10)
            plt.grid(True)
            plt.title(f"AT9 - Config. 2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"h(t)_schroeder_{int(f_central)}Hz_{fname}.png")
            plt.show()
