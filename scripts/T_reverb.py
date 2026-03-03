import os
import glob
import numpy as np
from scipy.signal import butter
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import parametros

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio')
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]


def computar_schroeder(audio_cortado_pa, fs, f_central):
    filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)
    energy_pa2 = filtered_pa**2
    schroeder_pa2 = np.cumsum(energy_pa2[::-1])[::-1]
    schroeder_norm = schroeder_pa2 / schroeder_pa2[0]
    schroeder_db = 10 * np.log10(schroeder_norm + 1e-12)
    t_sch = np.arange(len(schroeder_db)) / fs
    return filtered_pa, schroeder_db, t_sch


def ajustar_t60(t_sch, schroeder_db):
    mask = (schroeder_db <= -5) & (schroeder_db >= -25)
    t_fit = t_sch[mask]
    y_fit = schroeder_db[mask]
    if len(t_fit) <= 1:
        return None
    coeffs = np.polyfit(t_fit, y_fit, 1)
    T20 = (-20 - coeffs[1]) / coeffs[0]
    T30 = (-30 - coeffs[1]) / coeffs[0]
    T60 = (-60 - coeffs[1]) / coeffs[0]
    return coeffs, t_fit, T20, T30, T60


def plotar_schroeder_com_fd(t_sch, schroeder_db, coeffs, t_fit, T60, FD, f_central, titulo, savepath=None):
    is_125 = (f_central == 125)
    xlim_max = 2.0 if is_125 else 1.1
    x_arrow = 1.75 if is_125 else 0.9
    fator_extensao = 2.0
    t_sch_extended = np.linspace(0, np.max(t_sch) * fator_extensao, int(len(t_sch) * fator_extensao))
    line_full_extended = np.polyval(coeffs, t_sch_extended)

    plt.figure()
    plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
    plt.plot(t_sch_extended, line_full_extended, 'k--', label="Ajuste linear", linewidth=2)
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
    plt.text(x_arrow + 0.08, -FD/2, "Faixa \n dinâmica", ha='center', color='black', fontsize=9)
    plt.hlines(y=-5, xmin=0, xmax=x_arrow, color='gray', linestyle='--')
    plt.hlines(y=-FD, xmin=0, xmax=x_arrow, color='gray', linestyle='--')
    plt.annotate("", xy=(x_arrow, -FD-15), xytext=(x_arrow, -FD),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    plt.text(x_arrow + 0.08, -FD-7.5, "15 dB", ha='center', color='black', fontsize=9)
    plt.hlines(y=-FD-15, xmin=0, xmax=x_arrow, color='gray', linestyle='--')
    plt.annotate("", xy=(x_arrow, -90), xytext=(x_arrow, -FD-15),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    plt.text(x_arrow + 0.08, -FD-27, "Ruído de \n fundo", ha='center', color='black', fontsize=9)
    plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                 arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
    plt.text((t_fit[0]+t_fit[-1])/3, -59,
             rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)

    if savepath:
        plt.savefig(savepath)
    plt.show()


def processar_sala(impulse_files, noise_file, duracao_s, f_central_lista):
    _, audio_f1 = wav.read(noise_file)
    audio_f_pa = parametros.audio_pascal(audio_f1)

    for filepath in impulse_files:
        fname = os.path.splitext(os.path.basename(filepath))[0]
        fs, audio1 = wav.read(filepath)
        audio_pa = parametros.audio_pascal(audio1)
        audio_norm = audio1 / np.max(np.abs(audio1))

        NPS_m = parametros.L_max(audio_pa, 20e-6)
        r_f_global = parametros.L_equivalente(audio_f_pa, 20e-6)
        print(f"NPS máx.: {NPS_m:.2f} dB  |  Ruído de fundo: {r_f_global:.2f} dB")

        max_pos = np.unravel_index(np.argmax(np.abs(audio_norm)), audio_norm.shape)
        soma_indice = int(duracao_s * fs) - 1
        audio_cortado_pa = audio_pa[max_pos[0] : max_pos[0] + soma_indice]
        t_cortado = np.arange(len(audio_cortado_pa)) / fs

        for f_central in f_central_lista:
            filtered_pa, schroeder_db, _ = computar_schroeder(audio_cortado_pa, fs, f_central)
            t_sch = t_cortado
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
            print(f"Banda {f_central} Hz: T20={T20:.2f}s  T30={T30:.2f}s  T60={T60:.2f}s")


if __name__ == "__main__":
    audio = parametros.audio
    fs = parametros.fs
    t = parametros.t

    max_pos_linear = np.argmax(audio)
    max_pos = np.unravel_index(max_pos_linear, audio.shape)
    soma_indice = 28230
    indice_inicio = max_pos[0] - 600
    indice_fim = max_pos[0] + soma_indice
    t_inicio = t[indice_inicio]
    t_fim = t[indice_fim]
    audio_cortado = audio[indice_inicio:indice_fim]
    t_cortado = np.arange(t_inicio, t_fim, 1/fs)

    for f_central in f_central_lista:
        filtered = parametros.filtro_banda(audio_cortado, fs, f_central)

        energy = filtered**2
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder /= np.max(schroeder)
        schroeder_db = 10 * np.log10(schroeder)
        t_sch = np.arange(len(schroeder_db)) / fs

        energy_db = 10 * np.log10(energy / np.max(energy))

        mask = (schroeder_db <= 0) & (schroeder_db >= -25)
        t_fit = t_sch[mask]
        y_fit = schroeder_db[mask]

        if len(t_fit) > 1:
            coeffs = np.polyfit(t_fit, y_fit, 1)
            T20 = -20 / coeffs[0]
            T30 = -30 / coeffs[0]
            T60 = -60 / coeffs[0]
            line_fit = np.polyval(coeffs, t_fit)
            line_full = np.polyval(coeffs, t_sch)

            plt.figure()
            plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
            plt.plot(t_sch, line_full, 'k--', label="Ajuste linear", linewidth=2)
            plt.xlabel("Tempo [s]")
            plt.ylabel("Nível [dB]")
            plt.title(f"{f_central} Hz")
            plt.grid(True)
            plt.xlim(0, 1.2)
            plt.ylim(-100, 10)
            plt.legend()
            plt.tight_layout()
            plt.annotate("", xy=(0, -20), xytext=(T20, -20),
                         arrowprops=dict(arrowstyle='<->', color='grey', lw=1.5))
            plt.text((t_fit[0]+t_fit[-1])/3, -19, f"T20 ≈ {T20:.2f} s", ha='center', color='Black')

            plt.annotate("", xy=(0, -30), xytext=(T30, -30),
                         arrowprops=dict(arrowstyle='<->', color='grey', lw=1.5))
            plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')

            plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                         arrowprops=dict(arrowstyle='<->', color='grey', lw=1.5))
            plt.text((t_fit[0]+t_fit[-1])/2, -59, f"T60 ≈ {T60:.2f} s", ha='center', color='Black')

            plt.show()

            print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
            print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
            print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
        else:
            print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")

    r_f = 44.73

    max_pos_linear = np.argmax(audio)
    max_pos = np.unravel_index(max_pos_linear, audio.shape)
    soma_indice = 28230
    indice_inicio = max_pos[0] - 600
    indice_fim = max_pos[0] + soma_indice
    t_inicio = t[indice_inicio]
    t_fim = t[indice_fim]
    audio_cortado = audio[indice_inicio:indice_fim]
    t_cortado = np.arange(t_inicio, t_fim, 1/fs)

    for f_central in f_central_lista:
        filtered = parametros.filtro_banda(audio_cortado, fs, f_central)

        energy = filtered**2
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder /= np.max(schroeder)
        schroeder_db = 10 * np.log10(schroeder)
        t_sch = np.arange(len(schroeder_db)) / fs

        energy_db = 10 * np.log10(energy / np.max(energy))

        mask = (schroeder_db <= -5) & (schroeder_db >= -(25))
        t_fit = t_sch[mask]
        y_fit = schroeder_db[mask]
        if f_central == 125:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = -(20 / coeffs[0])*3
                T30 = -(30 / coeffs[0])*2
                T60 = -60 / coeffs[0]
                line_fit = np.polyval(coeffs, t_fit)
                line_full = np.polyval(coeffs, t_sch)

                fator_extensao = 2.0
                t_max_original = np.max(t_sch)
                t_max_estendido = t_max_original * fator_extensao

                t_sch_extended = np.linspace(0, t_max_estendido, int(len(t_sch) * fator_extensao))
                line_full_extended = np.polyval(coeffs, t_sch_extended)

                plt.figure()
                plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
                plt.plot(t_sch_extended, line_full_extended, 'k--', label="Ajuste linear", linewidth=2)
                plt.xlabel("Tempo [s]")
                plt.ylabel("NPS [dB]")
                plt.title(f"{f_central} Hz")
                plt.grid(True)
                plt.xlim(0, 2.0)
                plt.ylim(-100, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2])
                plt.legend()
                plt.tight_layout()

                plt.annotate("", xy=(1.75, -(r_f-15)), xytext=(1.75, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.87, -((r_f/2)-2), "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=1.75, color='gray', linestyle='--')
                plt.hlines(y=-(r_f-15), xmin=0, xmax=1.75, color='gray', linestyle='--')

                plt.annotate("", xy=(1.75, -r_f), xytext=(1.75, -(r_f-15)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.86, -((r_f/2)+15), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=-r_f, xmin=0, xmax=1.75, color='gray', linestyle='--')

                plt.annotate("", xy=(1.75, -100), xytext=(1.75, -r_f),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.88, -((r_f/2)+37), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} \approx {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)

                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
        else:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = -(20 / coeffs[0])*3
                T30 = -(30 / coeffs[0])*2
                T60 = -60 / coeffs[0]
                line_fit = np.polyval(coeffs, t_fit)
                line_full = np.polyval(coeffs, t_sch)

                plt.figure()
                plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="orange")
                plt.plot(t_sch, line_full, 'k--', label="Ajuste linear", linewidth=2)
                plt.xlabel("Tempo [s]")
                plt.ylabel("NPS [dB]")
                plt.title(f"{f_central} Hz")
                plt.grid(True)
                plt.xlim(0, 1.2)
                plt.ylim(-100, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2])
                plt.yticks(range(-100, 0, 10))
                plt.legend()
                plt.tight_layout()

                plt.annotate("", xy=(1, -(r_f-15)), xytext=(1, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.08, -((r_f/2)-2), "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=1, color='gray', linestyle='--')
                plt.hlines(y=-(r_f-15), xmin=0, xmax=1, color='gray', linestyle='--')

                plt.annotate("", xy=(1, -r_f), xytext=(1, -(r_f-15)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.08, -((r_f/2)+15), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=-r_f, xmin=0, xmax=1, color='gray', linestyle='--')

                plt.annotate("", xy=(1, -100), xytext=(1, -r_f),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.08, -((r_f/2)+37), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} \approx {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)

                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")

    at10_files = sorted([f for f in glob.glob(os.path.join(AUDIO_DIR, 'AT10_impulse_[0-9]*.wav'))
                         if 'nao_usado' not in f])
    at5_files  = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT5_impulse_[0-9]*.wav')))
    at9_files  = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT9_impulse_[0-9]*.wav')))

    processar_sala(at10_files, os.path.join(AUDIO_DIR, 'AT10_noise_background_1.wav'), 0.8, f_central_lista)
    processar_sala(at5_files,  os.path.join(AUDIO_DIR, 'AT5_noise_background_mic1.wav'), 0.8, f_central_lista)
    processar_sala(at9_files,  os.path.join(AUDIO_DIR, 'AT9_noise_background_1.wav'), 1.8, f_central_lista)
