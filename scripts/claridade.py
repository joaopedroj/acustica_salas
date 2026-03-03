# Todas as frequencias de banda de oitava (Schroeder + ajuste)

import os
import glob
import numpy as np
from scipy.signal import butter
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import parametros

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio')

if __name__ == "__main__":
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
        print(f"Banda {f_central} Hz: D50 ≈ {D50*100:.3f}%")
        print(f"Banda {f_central} Hz: C50 ≈ {C50:.3f}")

        # # Plot 3: Energia instantânea
        # energy_db = 10 * np.log10(energy / np.max(energy))
        # plt.figure()
        # plt.plot(t_sch, energy_db, label="Energia instantânea", alpha=0.5, color="dodgerblue")
        # plt.xlabel("Tempo [s]")
        # plt.ylabel("Nível [dB]")
        # plt.xlim(0, 1.2)
        # plt.ylim(-120, 10)
        # plt.grid(True)
        # plt.title(f"{f_central} Hz")
        # plt.legend()
        # plt.tight_layout()
        # # plt.savefig(f"h(t)_schroeder_{int(f_central)}Hz.png")
        # plt.show()

    # Discover AT5, AT9, AT10 impulse files via glob
    at9_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT9_impulse_[0-9]*.wav')))
    at10_files = sorted([f for f in glob.glob(os.path.join(AUDIO_DIR, 'AT10_impulse_[0-9]*.wav'))
                         if 'nao_usado' not in f])
    at5_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT5_impulse_[0-9]*.wav')))

    # Frequências centrais das bandas de oitava
    f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
    fs_target = 51200
    Lista_ATs = [5, 9, 10]

    for j in Lista_ATs:
        if j == 9:
            files = at9_files
        elif j == 10:
            files = at10_files
        else:
            files = at5_files

        for filepath in files:
            # Carregamento do .wav
            fs, audio = wav.read(filepath)
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

            fname = os.path.splitext(os.path.basename(filepath))[0]

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
                print(f"AT{j} {fname} - Banda {f_central} Hz: D50 ≈ {D50*100:.3f}%")
                print(f"AT{j} {fname} - Banda {f_central} Hz: C50 ≈ {C50:.3f}")

    # Apenas valores

    import numpy as np
    from scipy.signal import butter
    import scipy.signal as signal
    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plt

    import parametros


    # Frequências centrais das bandas de oitava
    f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
    fs_target = 51200
    Lista_ATs = [5, 9, 10]
    C50_AT5 = np.zeros((8, 7))
    C50_AT9 = np.zeros((8, 7))
    C50_AT10 = np.zeros((8, 7))

    for j in Lista_ATs:
        if j == 9:
            files = at9_files
        elif j == 10:
            files = at10_files
        else:
            files = at5_files

        for file_idx, filepath in enumerate(files):
            # Carregamento do .wav
            fs, audio = wav.read(filepath)
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
            aux = 0
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
                if j == 9:
                    C50_AT9[file_idx, aux] = C50
                elif j == 10:
                    C50_AT10[file_idx, aux] = C50
                else:
                    C50_AT5[file_idx, aux] = C50
                aux = aux + 1
                # d_50.append(D50)
                # print(f"AT{j} impulso {file_idx+1} - Banda {f_central} Hz: D50 ≈ {D50*100:.3f}%")
                # print(f"AT{j} impulso {file_idx+1} - Banda {f_central} Hz: C50 ≈ {C50:.3f}")

    print(C50_AT5)
    print(C50_AT9)
    print(C50_AT10)

    # Calcular a média de cada linha
    media_AT5 = np.mean(C50_AT5, axis=0)
    media_AT9 = np.mean(C50_AT9, axis=0)
    media_AT10 = np.mean(C50_AT10, axis=0)

    # Exibir os resultados
    print("Média de cada linha de C50_AT5:")
    print(media_AT5)
    print(f"Shape: {media_AT5.shape}")

    print("\nMédia de cada linha de C50_AT9:")
    print(media_AT9)
    print(f"Shape: {media_AT9.shape}")

    print("\nMédia de cada linha de C50_AT10:")
    print(media_AT10)
    print(f"Shape: {media_AT10.shape}")

    teste=[-7.68301103, -9.29175173, -9.73770402, -8.80081669, -6.98500914, -9.31168588, -10.3770543, -12.7441116]
    print(np.mean(teste))
