# Plot do T60 sem faixa dinâmica

# %%
import numpy as np
from scipy.signal import butter
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import parametros
audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t

# --- Definição da janela ---
max_pos_linear = np.argmax(audio)
max_pos = np.unravel_index(max_pos_linear, audio.shape)
soma_indice = 28230
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
t_inicio = t[indice_inicio]
t_fim = t[indice_fim]
audio_cortado = audio[indice_inicio:indice_fim]
t_cortado = np.arange(t_inicio, t_fim, 1/fs)


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


    energy_db = 10 * np.log10(energy / np.max(energy))

    # Plot 4: Ajuste linear para T20
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
        # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")
        # Adiciona setas ilustrativas para T20, T30 e T60
        plt.annotate("", xy=(0, -20), xytext=(T20, -20),
                     arrowprops=dict(arrowstyle='<->', color='grey', lw=1.5))
        plt.text((t_fit[0]+t_fit[-1])/3, -19, f"T20 ≈ {T20:.2f} s", ha='center', color='Black')
        
        plt.annotate("", xy=(0, -30), xytext=(T30, -30),
                     arrowprops=dict(arrowstyle='<->', color='grey', lw=1.5))
        plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')
        
        plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                     arrowprops=dict(arrowstyle='<->', color='grey', lw=1.5))
        plt.text((t_fit[0]+t_fit[-1])/2, -59, f"T60 ≈ {T60:.2f} s", ha='center', color='Black')
        
        # plt.savefig(f"T20_T30_T60_{int(f_central)}Hz.png")
        plt.show()

        print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
        print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
        print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
    else:
        print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")

# %%
# Plot do T60 com faixa dinâmica
import numpy as np
from scipy.signal import butter
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import parametros
audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t

r_f = 44.73

# --- Definição da janela ---
max_pos_linear = np.argmax(audio)
max_pos = np.unravel_index(max_pos_linear, audio.shape)
soma_indice = 28230
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
t_inicio = t[indice_inicio]
t_fim = t[indice_fim]
audio_cortado = audio[indice_inicio:indice_fim]
t_cortado = np.arange(t_inicio, t_fim, 1/fs)


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


    energy_db = 10 * np.log10(energy / np.max(energy))

    # Plot 4: Ajuste linear para T20
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
            # plt.yticks(range(-100, 0, 10))
            plt.legend()
            plt.tight_layout()
            # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

            # Adiciona setas para marcação da faixa dinâmica
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
            

            # plt.savefig(f"T20_T30_T60_{int(f_central)}Hz_3.png")
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
            # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

            # Adiciona setas para marcação da faixa dinâmica
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

            # Adiciona setas ilustrativas para T20, T30 e T60        

            # plt.annotate("", xy=(0, -20), xytext=(T20/3, -20),
            #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
            # plt.text((t_fit[0]+t_fit[-1])/3, -19, rf"$T_{{20}}/3 \approx {T20/3:.2f}\,\mathrm{{s}}$", ha='center', color='Black',
            #          fontsize=9)
            
            # plt.annotate("", xy=(0, -30), xytext=(T30, -30),
            #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
            # plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')
            
            plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                        arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
            plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} \approx {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
            

            # plt.savefig(f"T20_T30_T60_{int(f_central)}Hz_2.png")
            plt.show()

            print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
            print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
            print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
        else:
            print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")

# %%
##################################################################################################################
# automatização
import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

# Frequências centrais das bandas de oitava
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200  # Frequência de amostragem típica do sonômetro

r_f = 44.73

for i in range(1,2):

    fs, audio1 = wav.read(f"AT10_explosao{i}.wav")
    audio = audio1 / np.max(np.abs(audio1))  # Normalização
    t = np.arange(0, (len(audio)-1)/fs + 1/fs, 1/fs)

    fs, audio_f1 = wav.read(f"AT10_ruido fundo1.wav")
    audio_f = audio_f1 / np.max(np.abs(audio_f1))  # Normalização
    t = np.arange(0, (len(audio_f)-1)/fs + 1/fs, 1/fs)

    NPS_m = parametros.L_max(parametros.audio_pascal(audio1), 20e-6) # Nível máximo do ruído impulsivo
    print(f"Nível de pressão sonora máx.: {NPS_m}")

    # --- Definição da janela ---
    max_pos_linear = np.argmax(audio)
    max_pos = np.unravel_index(max_pos_linear, audio.shape)
    # soma_indice = int(2.3*fs) - 1
    soma_indice = int(1*fs) - 1
    indice_inicio = max_pos[0]
    indice_fim = max_pos[0] + soma_indice
    t_inicio = t[indice_inicio]
    t_fim = t[indice_fim]
    audio_cortado = audio[indice_inicio:indice_fim]

    # Criação do vetor de tempo baseado no número real de amostras
    n_amostras = len(audio_cortado)
    t_cortado = np.arange(n_amostras) / fs  # Tempo relativo começando em 0


    # --- Loop para cada banda de oitava ---
    for f_central in f_central_lista:
        # Filtragem
        filtered = parametros.filtro_banda(audio_cortado, fs, f_central)
        print(f"Ruído de fundo ({f_central} - {i}): {r_f}")

        FD = NPS_m - 15 - r_f # Faixa dinâmica

        # Curva de Schroeder (Resposta ao Impulso)
        energy = filtered**2
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder /= np.max(schroeder)
        schroeder_db = 10 * np.log10(schroeder + 1e-12)  # Evita log(0)
        # Usa o MESMO vetor de tempo
        t_sch = t_cortado  # Garante correspondência perfeita

        energy_db = 10 * np.log10(energy / np.max(energy))

        # Plot 4: Ajuste linear para T20
        mask = (schroeder_db <= -5) & (schroeder_db >= -25)
        t_fit = t_sch[mask]
        y_fit = schroeder_db[mask]
        if f_central == 125:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                # plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2])
                plt.yticks(range(-100, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(1.75, -(FD)), xytext=(1.75, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.87, -((r_f/2)-4), "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=1.75, color='gray', linestyle='--')
                plt.hlines(y=-(FD), xmin=0, xmax=1.75, color='gray', linestyle='--')

                plt.annotate("", xy=(1.75, -r_f), xytext=(1.75, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.86, -((r_f/2)+11), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=-r_f, xmin=0, xmax=1.75, color='gray', linestyle='--')

                plt.annotate("", xy=(1.75, -100), xytext=(1.75, -r_f),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.88, -((r_f/2)+33), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                plt.savefig(f"T60_{int(f_central)}Hz_AT10_{i}.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
        else:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                plt.yticks(range(-100, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(1, -(FD)), xytext=(1, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.08, -((r_f/2)-2), "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=1, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=1, color='gray', linestyle='--')

                plt.annotate("", xy=(1, -FD-15), xytext=(1, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.08, -((r_f/2)+15), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=1, color='gray', linestyle='--')

                plt.annotate("", xy=(1, -100), xytext=(1, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(1.08, -((r_f/2)+37), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                # Adiciona setas ilustrativas para T20, T30 e T60        

                # plt.annotate("", xy=(0, -20), xytext=(T20/3, -20),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/3, -19, rf"$T_{{20}}/3 = {T20/3:.2f}\,\mathrm{{s}}$", ha='center', color='Black',
                #          fontsize=9)
                
                # plt.annotate("", xy=(0, -30), xytext=(T30, -30),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')
                
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                plt.savefig(f"T60_{int(f_central)}Hz_AT10_{i}.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
# %%

# %%
##################################################################################################################
# automatização 2 AT10



import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

# Frequências centrais das bandas de oitava
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200

for i in range(1, 9):
    # ===== ÁUDIO DA EXPLOSÃO =====
    fs, audio1 = wav.read(f"AT10_explosao{i}.wav")
    audio_pa = parametros.audio_pascal(audio1)  # Converte para Pascal

    # # ===== ÁUDIO DA EXPLOSÃO =====
    # fs, audio1 = wav.read(f"AT5_explosão{i}.wav")
    # audio_pa = parametros.audio_pascal(audio1)  # Converte para Pascal
    
    # Normaliza apenas para localizar o pico (não usa para cálculos!)
    audio_norm = audio1 / np.max(np.abs(audio1))
    t = np.arange(len(audio_pa)) / fs

    # # ===== ÁUDIO DO RUÍDO DE FUNDO =====
    # fs, audio_f1 = wav.read(f"AT10_ruido fundo1.wav")
    # audio_f_pa = parametros.audio_pascal(audio_f1)  # Converte para Pascal


    # ===== ÁUDIO DO RUÍDO DE FUNDO =====
    fs, audio_f1 = wav.read(f"AT10_ruido fundo1.wav")
    audio_f_pa = parametros.audio_pascal(audio_f1)  # Converte para Pascal

    # ===== CÁLCULOS GLOBAIS =====
    NPS_m = parametros.L_max(audio_pa, 20e-6)
    print(f"Nível de pressão sonora máx.: {NPS_m:.2f} dB")

    r_f_global = parametros.L_equivalente(audio_f_pa, 20e-6)
    print(f"Ruído de fundo sem filtro: {r_f_global:.2f} dB")

    # ===== DEFINIÇÃO DA JANELA (usa normalizado só para encontrar posição) =====
    max_pos_linear = np.argmax(np.abs(audio_norm))
    max_pos = np.unravel_index(max_pos_linear, audio_norm.shape)
    soma_indice = int(0.8*fs) - 1
    indice_inicio = max_pos[0]
    indice_fim = max_pos[0] + soma_indice
    
    # Corta o áudio EM PASCAL
    audio_cortado_pa = audio_pa[indice_inicio:indice_fim]
    
    n_amostras = len(audio_cortado_pa)
    t_cortado = np.arange(n_amostras) / fs

    # ===== LOOP POR BANDA =====
    for f_central in f_central_lista:
        # Filtra os sinais EM PASCAL
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)
        filtered_fundo_pa = parametros.filtro_banda(audio_f_pa, fs, f_central)

        NPS_mf = parametros.L_max(filtered_pa, 20e-6)
        # print(f"Nível de pressão sonora máx.{f_central} Hz - {i}): {NPS_mf:.2f} dB")

        r_f = parametros.L_equivalente(filtered_fundo_pa, 20e-6)
        # print(f"Ruído de fundo ({f_central} Hz - {i}): {r_f:.2f} dB")

        FD = NPS_mf - 15 - r_f - 5 # Faixa dinâmica

        # Filtrar em Pascal
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)

        # Energia em Pa²
        energy_pa2 = filtered_pa**2

        # Curva de Schroeder (integração reversa)
        schroeder_pa2 = np.cumsum(energy_pa2[::-1])[::-1]

        # Normalizar pela energia máxima (início da curva)
        schroeder_norm = schroeder_pa2 / schroeder_pa2[0]

        # Converter para dB (agora começa em 0 dB e decai)
        schroeder_db = 10 * np.log10(schroeder_norm + 1e-12)

        # Energia instantânea normalizada em dB
        energy_norm = energy_pa2 / np.max(energy_pa2)
        energy_db = 10 * np.log10(energy_norm + 1e-12)

        # Usa o MESMO vetor de tempo
        t_sch = t_cortado  # Garante correspondência perfeita


        # Plot 4: Ajuste linear para T20
        mask = (schroeder_db <= -5) & (schroeder_db >= -25)
        t_fit = t_sch[mask]
        y_fit = schroeder_db[mask]
        if f_central == 125:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                plt.xlim(0, 1.25)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(0.9, -(FD)), xytext=(0.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, -FD/2, "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=0.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -FD-15), xytext=(0.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -90), xytext=(0.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=9)
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                # plt.savefig(f"T60_{int(f_central)}Hz_AT10_{i}.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Impulso {i} - Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
        else:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                plt.xlim(0, 1.1)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(0.9, -(FD)), xytext=(0.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, -FD/2, "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=0.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -FD-15), xytext=(0.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -90), xytext=(0.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                # Adiciona setas ilustrativas para T20, T30 e T60        

                # plt.annotate("", xy=(0, -20), xytext=(T20/3, -20),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/3, -19, rf"$T_{{20}}/3 = {T20/3:.2f}\,\mathrm{{s}}$", ha='center', color='Black',
                #          fontsize=9)
                
                # plt.annotate("", xy=(0, -30), xytext=(T30, -30),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')
                
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                # plt.savefig(f"T60_{int(f_central)}Hz_AT10_{i}.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Impulso {i} - Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
# %%

# %%
##################################################################################################################
# automatização 2 AT5



import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

# Frequências centrais das bandas de oitava
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200

for i in range(1, 9):
    # ===== ÁUDIO DA EXPLOSÃO =====
    fs, audio1 = wav.read(f"AT5_explosão{i}.wav")
    audio_pa = parametros.audio_pascal(audio1)  # Converte para Pascal
    
    # Normaliza apenas para localizar o pico (não usa para cálculos!)
    audio_norm = audio1 / np.max(np.abs(audio1))
    t = np.arange(len(audio_pa)) / fs

    # ===== ÁUDIO DO RUÍDO DE FUNDO =====
    fs, audio_f1 = wav.read(f"AT5_ruido fundo mic1.wav")
    audio_f_pa = parametros.audio_pascal(audio_f1)  # Converte para Pascal

    # ===== CÁLCULOS GLOBAIS =====
    NPS_m = parametros.L_max(audio_pa, 20e-6)
    print(f"Nível de pressão sonora máx.: {NPS_m:.2f} dB")

    r_f_global = parametros.L_equivalente(audio_f_pa, 20e-6)
    print(f"Ruído de fundo sem filtro: {r_f_global:.2f} dB")

    # ===== DEFINIÇÃO DA JANELA (usa normalizado só para encontrar posição) =====
    max_pos_linear = np.argmax(np.abs(audio_norm))
    max_pos = np.unravel_index(max_pos_linear, audio_norm.shape)
    soma_indice = int(0.8*fs) - 1
    indice_inicio = max_pos[0]
    indice_fim = max_pos[0] + soma_indice
    
    # Corta o áudio EM PASCAL
    audio_cortado_pa = audio_pa[indice_inicio:indice_fim]
    
    n_amostras = len(audio_cortado_pa)
    t_cortado = np.arange(n_amostras) / fs

    # ===== LOOP POR BANDA =====
    for f_central in f_central_lista:
        # Filtra os sinais EM PASCAL
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)
        filtered_fundo_pa = parametros.filtro_banda(audio_f_pa, fs, f_central)

        NPS_mf = parametros.L_max(filtered_pa, 20e-6)
        print(f"Nível de pressão sonora máx.{f_central} Hz - {i}): {NPS_mf:.2f} dB")

        r_f = parametros.L_equivalente(filtered_fundo_pa, 20e-6)
        print(f"Ruído de fundo ({f_central} Hz - {i}): {r_f:.2f} dB")

        FD = NPS_mf - 15 - r_f - 5 # Faixa dinâmica
        print(FD)

        # Filtrar em Pascal
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)

        # Energia em Pa²
        energy_pa2 = filtered_pa**2

        # Curva de Schroeder (integração reversa)
        schroeder_pa2 = np.cumsum(energy_pa2[::-1])[::-1]

        # Normalizar pela energia máxima (início da curva)
        schroeder_norm = schroeder_pa2 / schroeder_pa2[0]

        # Converter para dB (agora começa em 0 dB e decai)
        schroeder_db = 10 * np.log10(schroeder_norm + 1e-12)

        # Energia instantânea normalizada em dB
        energy_norm = energy_pa2 / np.max(energy_pa2)
        energy_db = 10 * np.log10(energy_norm + 1e-12)

        # Usa o MESMO vetor de tempo
        t_sch = t_cortado  # Garante correspondência perfeita


        # Plot 4: Ajuste linear para T20
        mask = (schroeder_db <= -5) & (schroeder_db >= -25)
        t_fit = t_sch[mask]
        y_fit = schroeder_db[mask]
        if f_central == 125:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                # plt.title(f"{f_central} Hz")
                plt.title(f"AT5")
                plt.grid(True)
                plt.xlim(0, 1.25)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(0.9, -(FD)), xytext=(0.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, -FD/2, "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=0.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -FD-15), xytext=(0.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -90), xytext=(0.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=9)
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                plt.savefig(f"T60_{int(f_central)}Hz_AT5_{i}_titulo.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
        else:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                # plt.title(f"{f_central} Hz")
                plt.title(f"AT5")
                plt.grid(True)
                plt.xlim(0, 1.1)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(0.9, -(FD)), xytext=(0.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, -FD/2, "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=0.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -FD-15), xytext=(0.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -90), xytext=(0.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                # Adiciona setas ilustrativas para T20, T30 e T60        

                # plt.annotate("", xy=(0, -20), xytext=(T20/3, -20),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/3, -19, rf"$T_{{20}}/3 = {T20/3:.2f}\,\mathrm{{s}}$", ha='center', color='Black',
                #          fontsize=9)
                
                # plt.annotate("", xy=(0, -30), xytext=(T30, -30),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')
                
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                plt.savefig(f"T60_{int(f_central)}Hz_AT5_{i}_titulo.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
# %%

# %%
##################################################################################################################
# automatização 2 AT9



import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

# Frequências centrais das bandas de oitava
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200

for i in range(1, 9):
    # ===== ÁUDIO DA EXPLOSÃO =====
    fs, audio1 = wav.read(f"AT9_explosão{i}.wav")
    audio_pa = parametros.audio_pascal(audio1)  # Converte para Pascal
    
    # Normaliza apenas para localizar o pico (não usa para cálculos!)
    audio_norm = audio1 / np.max(np.abs(audio1))
    t = np.arange(len(audio_pa)) / fs

    # ===== ÁUDIO DO RUÍDO DE FUNDO =====
    fs, audio_f1 = wav.read(f"AT9_ruido fundo.wav")
    audio_f_pa = parametros.audio_pascal(audio_f1)  # Converte para Pascal

    # ===== CÁLCULOS GLOBAIS =====
    NPS_m = parametros.L_max(audio_pa, 20e-6)
    print(f"Nível de pressão sonora máx.: {NPS_m:.2f} dB")

    r_f_global = parametros.L_equivalente(audio_f_pa, 20e-6)
    print(f"Ruído de fundo sem filtro: {r_f_global:.2f} dB")

    # ===== DEFINIÇÃO DA JANELA (usa normalizado só para encontrar posição) =====
    max_pos_linear = np.argmax(np.abs(audio_norm))
    max_pos = np.unravel_index(max_pos_linear, audio_norm.shape)
    soma_indice = int(1.8*fs) - 1
    indice_inicio = max_pos[0]
    indice_fim = max_pos[0] + soma_indice
    
    # Corta o áudio EM PASCAL
    audio_cortado_pa = audio_pa[indice_inicio:indice_fim]
    
    n_amostras = len(audio_cortado_pa)
    t_cortado = np.arange(n_amostras) / fs

    # ===== LOOP POR BANDA =====
    for f_central in f_central_lista:
        # Filtra os sinais EM PASCAL
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)
        filtered_fundo_pa = parametros.filtro_banda(audio_f_pa, fs, f_central)

        NPS_mf = parametros.L_max(filtered_pa, 20e-6)
        print(f"Nível de pressão sonora máx.{f_central} Hz - {i}): {NPS_mf:.2f} dB")

        r_f = parametros.L_equivalente(filtered_fundo_pa, 20e-6)
        print(f"Ruído de fundo ({f_central} Hz - {i}): {r_f:.2f} dB")

        FD = NPS_mf - 15 - r_f - 5 # Faixa dinâmica
        print(FD)

        # Filtrar em Pascal
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)

        # Energia em Pa²
        energy_pa2 = filtered_pa**2

        # Curva de Schroeder (integração reversa)
        schroeder_pa2 = np.cumsum(energy_pa2[::-1])[::-1]

        # Normalizar pela energia máxima (início da curva)
        schroeder_norm = schroeder_pa2 / schroeder_pa2[0]

        # Converter para dB (agora começa em 0 dB e decai)
        schroeder_db = 10 * np.log10(schroeder_norm + 1e-12)

        # Energia instantânea normalizada em dB
        energy_norm = energy_pa2 / np.max(energy_pa2)
        energy_db = 10 * np.log10(energy_norm + 1e-12)

        # Usa o MESMO vetor de tempo
        t_sch = t_cortado  # Garante correspondência perfeita


        # Plot 4: Ajuste linear para T20
        mask = (schroeder_db <= -5) & (schroeder_db >= -25)
        t_fit = t_sch[mask]
        y_fit = schroeder_db[mask]
        if f_central == 125:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                # plt.title(f"{f_central} Hz")
                plt.title(f"AT9 - Config. 1")
                plt.grid(True)
                plt.xlim(0, 3.0)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(1.9, -(FD)), xytext=(1.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(2.06, -FD/2, "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=1.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=1.9, color='gray', linestyle='--')

                plt.annotate("", xy=(1.9, -FD-15), xytext=(1.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(2.06, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=1.9, color='gray', linestyle='--')

                plt.annotate("", xy=(1.9, -90), xytext=(1.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(2.06, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)

                plt.savefig(f"T60_{int(f_central)}Hz_AT9_{i}_titulo.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
        else:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                # plt.title(f"{f_central} Hz")
                plt.title(f"AT9 - Config. 1")
                plt.grid(True)
                plt.xlim(0, 2.6)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(1.9, -(FD)), xytext=(1.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(2.06, -FD/2, "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=1.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=1.9, color='gray', linestyle='--')

                plt.annotate("", xy=(1.9, -FD-15), xytext=(1.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(2.06, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=1.9, color='gray', linestyle='--')

                plt.annotate("", xy=(1.9, -90), xytext=(1.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(2.06, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=9)

                # Adiciona setas ilustrativas para T20, T30 e T60        

                # plt.annotate("", xy=(0, -20), xytext=(T20/3, -20),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/3, -19, rf"$T_{{20}}/3 = {T20/3:.2f}\,\mathrm{{s}}$", ha='center', color='Black',
                #          fontsize=9)
                
                # plt.annotate("", xy=(0, -30), xytext=(T30, -30),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')
                
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                plt.savefig(f"T60_{int(f_central)}Hz_AT9_{i}_titulo.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
# %%


# %%
##################################################################################################################
# manipulação AT10 para a apresentação



import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import parametros

# Frequências centrais das bandas de oitava
f_central_lista = [4000, ]
fs_target = 51200

for i in range(7, 8):
    # ===== ÁUDIO DA EXPLOSÃO =====
    fs, audio1 = wav.read(f"AT10_explosao{i}.wav")
    audio_pa = parametros.audio_pascal(audio1)  # Converte para Pascal

    # # ===== ÁUDIO DA EXPLOSÃO =====
    # fs, audio1 = wav.read(f"AT5_explosão{i}.wav")
    # audio_pa = parametros.audio_pascal(audio1)  # Converte para Pascal
    
    # Normaliza apenas para localizar o pico (não usa para cálculos!)
    audio_norm = audio1 / np.max(np.abs(audio1))
    t = np.arange(len(audio_pa)) / fs

    # # ===== ÁUDIO DO RUÍDO DE FUNDO =====
    # fs, audio_f1 = wav.read(f"AT10_ruido fundo1.wav")
    # audio_f_pa = parametros.audio_pascal(audio_f1)  # Converte para Pascal


    # ===== ÁUDIO DO RUÍDO DE FUNDO =====
    fs, audio_f1 = wav.read(f"AT10_ruido fundo1.wav")
    audio_f_pa = parametros.audio_pascal(audio_f1)  # Converte para Pascal

    # ===== CÁLCULOS GLOBAIS =====
    NPS_m = parametros.L_max(audio_pa, 20e-6)
    print(f"Nível de pressão sonora máx.: {NPS_m:.2f} dB")

    r_f_global = parametros.L_equivalente(audio_f_pa, 20e-6)
    print(f"Ruído de fundo sem filtro: {r_f_global:.2f} dB")

    # ===== DEFINIÇÃO DA JANELA (usa normalizado só para encontrar posição) =====
    max_pos_linear = np.argmax(np.abs(audio_norm))
    max_pos = np.unravel_index(max_pos_linear, audio_norm.shape)
    soma_indice = int(0.8*fs) - 1
    indice_inicio = max_pos[0]
    indice_fim = max_pos[0] + soma_indice
    
    # Corta o áudio EM PASCAL
    audio_cortado_pa = audio_pa[indice_inicio:indice_fim]
    
    n_amostras = len(audio_cortado_pa)
    t_cortado = np.arange(n_amostras) / fs

    # ===== LOOP POR BANDA =====
    for f_central in f_central_lista:
        # Filtra os sinais EM PASCAL
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)
        filtered_fundo_pa = parametros.filtro_banda(audio_f_pa, fs, f_central)

        NPS_mf = parametros.L_max(filtered_pa, 20e-6)
        # print(f"Nível de pressão sonora máx.{f_central} Hz - {i}): {NPS_mf:.2f} dB")

        r_f = parametros.L_equivalente(filtered_fundo_pa, 20e-6)
        # print(f"Ruído de fundo ({f_central} Hz - {i}): {r_f:.2f} dB")

        FD = NPS_mf - 15 - r_f - 5 # Faixa dinâmica

        # Filtrar em Pascal
        filtered_pa = parametros.filtro_banda(audio_cortado_pa, fs, f_central)

        # Energia em Pa²
        energy_pa2 = filtered_pa**2

        # Curva de Schroeder (integração reversa)
        schroeder_pa2 = np.cumsum(energy_pa2[::-1])[::-1]

        # Normalizar pela energia máxima (início da curva)
        schroeder_norm = schroeder_pa2 / schroeder_pa2[0]

        # Converter para dB (agora começa em 0 dB e decai)
        schroeder_db = 10 * np.log10(schroeder_norm + 1e-12)

        # Energia instantânea normalizada em dB
        energy_norm = energy_pa2 / np.max(energy_pa2)
        energy_db = 10 * np.log10(energy_norm + 1e-12)

        # Usa o MESMO vetor de tempo
        t_sch = t_cortado  # Garante correspondência perfeita


        # Plot 4: Ajuste linear para T20
        mask = (schroeder_db <= -5) & (schroeder_db >= -25)
        t_fit = t_sch[mask]
        y_fit = schroeder_db[mask]
        if f_central == 125:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
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
                plt.xlim(0, 1.25)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                # plt.savefig(f"schroeder_ajuste_linear_{int(f_central)}Hz.png")

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(0.9, -(FD)), xytext=(0.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, -FD/2, "Faixa \n dinâmica", ha='center', color='black',fontsize=9)
                plt.hlines(y=-5, xmin=0, xmax=0.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -FD-15), xytext=(0.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=9)
                plt.hlines(y=(-FD-15), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -90), xytext=(0.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.98, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=9)
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=9)
                

                # plt.savefig(f"T60_{int(f_central)}Hz_AT10_{i}.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Impulso {i} - Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
        else:
            if len(t_fit) > 1:
                coeffs = np.polyfit(t_fit, y_fit, 1)
                T20 = (-20 - coeffs[1]) / coeffs[0]
                T30 = (-30 - coeffs[1]) / coeffs[0]
                T60 = (-60 - coeffs[1]) / coeffs[0]
                line_fit = np.polyval(coeffs, t_fit)
                line_full = np.polyval(coeffs, t_sch)

                fator_extensao = 1.1
                t_max_original = np.max(t_sch)
                t_max_estendido = t_max_original * fator_extensao

                t_sch_extended = np.linspace(0, t_max_estendido, int(len(t_sch) * fator_extensao))
                line_full_extended = np.polyval(coeffs, t_sch_extended)


                plt.figure()
                plt.plot(t_sch, schroeder_db, label="Curva de Schroeder", color="salmon", linewidth=3.5)
                plt.plot(t_sch_extended, line_full_extended, 'k--', label="Ajuste linear", linewidth=2)
                plt.xlabel("Tempo [s]")
                plt.ylabel("NPS [dB]")
                # plt.title(f"{f_central} Hz")
                plt.grid(True)
                plt.xlim(0, 1.1)
                plt.ylim(-90, 10)
                plt.xticks([0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])
                plt.yticks(range(-90, 10, 10))
                plt.legend()
                plt.tight_layout()
                

                # Adiciona setas para marcação da faixa dinâmica
                plt.annotate("", xy=(0.9, -(FD)), xytext=(0.9, -5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.997, -FD/2 -7, "Faixa \n dinâmica", ha='center', color='black',fontsize=14)
                plt.hlines(y=-5, xmin=0, xmax=0.9, color='gray', linestyle='--')
                plt.hlines(y=(-FD), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -FD-15), xytext=(0.9, -(FD)),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.997, (-FD-15/2), "15 dB", ha='center', color='black', fontsize=14)
                plt.hlines(y=(-FD-15), xmin=0, xmax=0.9, color='gray', linestyle='--')

                plt.annotate("", xy=(0.9, -90), xytext=(0.9, -FD-15),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
                plt.text(0.997, (-FD-27), "Ruído de \n fundo", ha='center', color='black', fontsize=14)

                # Adiciona setas ilustrativas para T20, T30 e T60        

                # plt.annotate("", xy=(0, -20), xytext=(T20/3, -20),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/3, -19, rf"$T_{{20}}/3 = {T20/3:.2f}\,\mathrm{{s}}$", ha='center', color='Black',
                #          fontsize=9)
                
                # plt.annotate("", xy=(0, -30), xytext=(T30, -30),
                #              arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                # plt.text((t_fit[0]+t_fit[-1])/2, -29, f"T30 ≈ {T30:.2f} s", ha='center', color='Black')
                
                plt.annotate("", xy=(0, -60), xytext=(T60, -60),
                            arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
                plt.text((t_fit[0]+t_fit[-1])/3, -59, rf"$T_{{60}} = {T60:.2f}\,\mathrm{{s}}$", ha='center', color='Black', fontsize=14)
                

                plt.savefig(f"T60_aesthetic_final.png")
                plt.show()

                print(f"Banda {f_central} Hz: T20 ≈ {T20:.2f} s")
                print(f"Banda {f_central} Hz: T30 ≈ {T30:.2f} s")
                print(f"Impulso {i} - Banda {f_central} Hz: T60 ≈ {T60:.2f} s")
            else:
                print(f"Banda {f_central} Hz: dados insuficientes para ajuste T20.")
# %%