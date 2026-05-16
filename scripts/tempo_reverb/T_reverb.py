"""
T_reverb.py — Cálculo de T20, T30 e T60 por banda de oitava via curva de
Schroeder, com truncamento pelo método de Lundeby (ISO 3382-2 Anexo).
"""

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
from parametros import ajustar_t60

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'audio')
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]


def computar_schroeder(audio_cortado_pa, fs, f_central):
    """
    Aplica filtro_banda, computa energia, trunca via Lundeby e calcula a curva
    de Schroeder normalizada (em dB).

    Returns:
        Tupla (filtered_pa, schroeder_db, t_sch). O comprimento dos arrays
        retornados pode ser menor que len(audio_cortado_pa) — Lundeby trunca.
    """
    filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)
    energy_pa2 = filtered_pa ** 2

    idx_trunc = parametros.lundeby_truncation_point(energy_pa2, fs)
    energy_trunc = energy_pa2[:idx_trunc]
    filtered_trunc = filtered_pa[:idx_trunc]

    schroeder_pa2 = np.cumsum(energy_trunc[::-1])[::-1]
    if schroeder_pa2[0] <= 0:
        # fallback degenerado: sinal todo zero ou negativo após filtragem
        return filtered_trunc, np.full_like(schroeder_pa2, -120.0), \
            np.arange(len(schroeder_pa2)) / fs
    schroeder_norm = schroeder_pa2 / schroeder_pa2[0]
    schroeder_db = 10 * np.log10(schroeder_norm + 1e-12)
    t_sch = np.arange(len(schroeder_db)) / fs
    return filtered_trunc, schroeder_db, t_sch


def plotar_schroeder_com_fd(t_sch, schroeder_db, coeffs, t_fit, T60, FD,
                            f_central, titulo, savepath=None):
    is_125 = (f_central == 125)
    xlim_max = 2.0 if is_125 else 1.1
    x_arrow = 1.75 if is_125 else 0.9
    fator_extensao = 2.0
    t_sch_extended = np.linspace(0, np.max(t_sch) * fator_extensao,
                                 int(len(t_sch) * fator_extensao))
    line_full_extended = np.polyval(coeffs, t_sch_extended)

    plt.figure()
    plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
    plt.plot(t_sch_extended, line_full_extended, 'k--',
             label="Ajuste linear", linewidth=2)
    plt.xlabel("Tempo [s]")
    plt.ylabel("NPS [dB]")
    plt.title(titulo)
    plt.grid(True)
    plt.xlim(0, xlim_max)
    plt.ylim(-90, 10)
    plt.yticks(range(-90, 10, 10))
    plt.legend()
    plt.tight_layout()

    plt.annotate("", xy=(x_arrow, -FD), xytext=(x_arrow, -5),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    plt.text(x_arrow + 0.08, -FD / 2, "Faixa \n dinâmica",
             ha='center', color='black', fontsize=9)
    plt.hlines(y=-5, xmin=0, xmax=x_arrow, color='gray', linestyle='--')
    plt.hlines(y=-FD, xmin=0, xmax=x_arrow, color='gray', linestyle='--')
    plt.annotate("", xy=(x_arrow, -FD - 15), xytext=(x_arrow, -FD),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    plt.text(x_arrow + 0.08, -FD - 7.5, "15 dB",
             ha='center', color='black', fontsize=9)
    plt.hlines(y=-FD - 15, xmin=0, xmax=x_arrow, color='gray', linestyle='--')
    plt.annotate("", xy=(x_arrow, -90), xytext=(x_arrow, -FD - 15),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    plt.text(x_arrow + 0.08, -FD - 27, "Ruído de \n fundo",
             ha='center', color='black', fontsize=9)
    plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                 arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
    plt.text((t_fit[0] + t_fit[-1]) / 3, -59,
             rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$",
             ha='center', color='Black', fontsize=9)

    if savepath:
        plt.savefig(savepath)
    plt.show()


def processar_sala(impulse_files, noise_file, duracao_s, f_central_lista):
    """
    Processa todos os impulsos de uma sala. duracao_s é apenas referência inicial;
    a janela é ampliada a 2x e o truncamento real fica a cargo do Lundeby.
    """
    _, audio_f1 = wav.read(noise_file)
    audio_f_pa = parametros.audio_pascal(audio_f1)

    for filepath in impulse_files:
        fname = os.path.splitext(os.path.basename(filepath))[0]
        fs, audio1 = wav.read(filepath)
        audio_pa = parametros.audio_pascal(audio1, fs)
        audio_norm = audio1 / np.max(np.abs(audio1))

        NPS_m = parametros.L_max(audio_pa, 20e-6)
        r_f_global = parametros.L_equivalente(audio_f_pa, 20e-6)
        print(f"NPS máx.: {NPS_m:.2f} dB  |  Ruído de fundo: {r_f_global:.2f} dB")

        max_pos = int(np.argmax(np.abs(audio_norm)))

        # janela ampla: Lundeby corta dentro de computar_schroeder
        soma_indice = min(len(audio_pa) - max_pos, int(2.0 * duracao_s * fs))
        audio_cortado_pa = audio_pa[max_pos: max_pos + soma_indice]

        for f_central in f_central_lista:
            filtered_pa, schroeder_db, t_sch = computar_schroeder(
                audio_cortado_pa, fs, f_central
            )
            filtered_fundo_pa = parametros.filtro_banda(audio_f_pa, fs, f_central)
            NPS_mf = parametros.L_max(filtered_pa, 20e-6)
            r_f = parametros.L_equivalente(filtered_fundo_pa, 20e-6)
            FD = NPS_mf - 15 - r_f - 5

            resultado = ajustar_t60(t_sch, schroeder_db)
            if resultado is None:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
                continue
            coeffs, t_fit, T20, T30, T60 = resultado

            savepath = f"T60_{int(f_central)}Hz_{fname}.png"
            plotar_schroeder_com_fd(t_sch, schroeder_db, coeffs, t_fit, T60, FD,
                                    f_central, f"{f_central} Hz", savepath)
            print(f"Banda {f_central} Hz: T20={T20:.2f}s  "
                  f"T30={T30:.2f}s  T60={T60:.2f}s")


if __name__ == "__main__":
    at10_files = sorted([
        f for f in glob.glob(os.path.join(AUDIO_DIR, 'AT10_impulse_[0-9]*.wav'))
        if 'nao_usado' not in f
    ])
    at5_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT5_impulse_[0-9]*.wav')))
    at9_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT9_impulse_[0-9]*.wav')))

    processar_sala(at10_files,
                   os.path.join(AUDIO_DIR, 'AT10_noise_background_1.wav'),
                   0.8, f_central_lista)
    processar_sala(at5_files,
                   os.path.join(AUDIO_DIR, 'AT5_noise_background_mic1.wav'),
                   0.8, f_central_lista)
    processar_sala(at9_files,
                   os.path.join(AUDIO_DIR, 'AT9_noise_background_1.wav'),
                   1.8, f_central_lista)
