"""claridade.py — Cálculo de C50 (claridade) e D50 (definição) por banda de oitava."""

import os
import glob
import numpy as np
import scipy.io.wavfile as wav

import parametros

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio')


def calcular_c50_d50(audio_cortado, fs, f_central):
    """
    Calcula C50 (dB) e D50 (fração) para uma banda específica.

    Args:
        audio_cortado: array 1D já recortado a partir do pico do impulso.
        fs: frequência de amostragem em Hz.
        f_central: frequência central da banda em Hz.

    Returns:
        Tupla (C50, D50). C50 em dB; D50 adimensional (0–1).
    """
    filtered = parametros.filtro_banda(audio_cortado, fs, f_central)
    energy = filtered ** 2

    limite_50ms = int(0.05 * fs)
    energia_pre50 = np.trapz(energy[:limite_50ms], dx=1 / fs)
    energia_pos50 = np.trapz(energy[limite_50ms:], dx=1 / fs)
    energia_total = energia_pre50 + energia_pos50

    if energia_total <= 0 or energia_pos50 <= 0:
        return float('nan'), float('nan')

    D50 = energia_pre50 / energia_total
    C50 = 10 * np.log10(energia_pre50 / energia_pos50)
    return C50, D50


def janela_adaptativa(audio, fs, duracao_max_s=2.0):
    """
    Recorta o sinal a partir do pico, com duração máxima `duracao_max_s` ou
    o que sobrar — o que for menor.
    """
    max_pos = int(np.argmax(np.abs(audio)))
    soma_indice = min(len(audio) - max_pos, int(duracao_max_s * fs))
    audio_cortado = audio[max_pos: max_pos + soma_indice]
    return audio_cortado, max_pos


if __name__ == "__main__":
    f_central_lista = parametros.f_central_lista

    _default = parametros.load_default_audio()
    audio_ref = _default['audio']
    fs_ref = _default['fs']

    audio_cortado, _ = janela_adaptativa(audio_ref, fs_ref)

    print("=== Sinal de referência ===")
    for f_central in f_central_lista:
        C50, D50 = calcular_c50_d50(audio_cortado, fs_ref, f_central)
        print(f"Banda {f_central} Hz: D50 ≈ {D50*100:.3f}%   C50 ≈ {C50:.3f} dB")

    at5_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT5_impulse_[0-9]*.wav')))
    at9_files = sorted(glob.glob(os.path.join(AUDIO_DIR, 'AT9_impulse_[0-9]*.wav')))
    at10_files = sorted([
        f for f in glob.glob(os.path.join(AUDIO_DIR, 'AT10_impulse_[0-9]*.wav'))
        if 'nao_usado' not in f
    ])

    Lista_ATs = [(5, at5_files), (9, at9_files), (10, at10_files)]
    C50_por_sala = {5: [], 9: [], 10: []}

    for at_id, files in Lista_ATs:
        for filepath in files:
            fs, audio = wav.read(filepath)
            audio = audio / np.max(np.abs(audio))
            audio_cortado, _ = janela_adaptativa(audio, fs)

            fname = os.path.splitext(os.path.basename(filepath))[0]
            c50_linha = []
            for f_central in f_central_lista:
                C50, D50 = calcular_c50_d50(audio_cortado, fs, f_central)
                c50_linha.append(C50)
                print(f"AT{at_id} {fname} - Banda {f_central} Hz: "
                      f"D50 ≈ {D50*100:.3f}%   C50 ≈ {C50:.3f} dB")
            C50_por_sala[at_id].append(c50_linha)

    for at_id in (5, 9, 10):
        if not C50_por_sala[at_id]:
            continue
        arr = np.array(C50_por_sala[at_id])
        media = np.mean(arr, axis=0)
        print(f"\nMédia C50 AT{at_id}: {media}")
        print(f"Shape: {media.shape}")
