#%%
import os
import sys
import glob
import numpy as np
import scipy.io.wavfile as wav

# Permite execução standalone — adiciona scripts/ ao sys.path para achar parametros.py
_PARENT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import parametros

_HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(_HERE, '..', '..', 'audio')
OUT_DIR = os.path.join(_HERE, 'resultados')
OUT_FILE = os.path.join(OUT_DIR, 'claridade_resumo.txt')

# (id, exclude_token)
SALAS = [('AT5', None), ('AT9', None), ('AT10', 'nao_usado')]


def _coletar_arquivos(at_id, exclude_token):
    pattern = os.path.join(AUDIO_DIR, f'{at_id}_impulse_[0-9]*.wav')
    files = sorted(glob.glob(pattern))
    if exclude_token:
        files = [f for f in files if exclude_token not in f]
    return files


def calcular_c50_d50(audio_cortado, fs, f_central, pico_idx=0):
    filtered = parametros.filtro_banda(audio_cortado, fs, f_central)
    energy = filtered ** 2

    limite_50ms = pico_idx + int(0.05 * fs)
    energia_pre50 = np.trapz(energy[pico_idx:limite_50ms], dx=1 / fs)
    energia_pos50 = np.trapz(energy[limite_50ms:], dx=1 / fs)
    energia_total = energia_pre50 + energia_pos50

    if energia_total <= 0 or energia_pos50 <= 0:
        return float('nan'), float('nan')

    D50 = energia_pre50 / energia_total
    C50 = 10 * np.log10(energia_pre50 / energia_pos50)
    return C50, D50


def janela_adaptativa(audio, fs, duracao_max_s=2.0, pre_pico_s=0.1):
    max_pos = int(np.argmax(np.abs(audio)))
    pre_amostras = min(max_pos, int(pre_pico_s * fs))
    inicio = max_pos - pre_amostras
    fim = min(len(audio), max_pos + int(duracao_max_s * fs))
    audio_cortado = audio[inicio:fim]
    pico_idx = max_pos - inicio
    return audio_cortado, pico_idx


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    f_central_lista = parametros.f_central_lista

    linhas = []

    # ----- Sinal de referência -----
    _default = parametros.load_default_audio()
    audio_ref = _default['audio']
    fs_ref = _default['fs']
    audio_cortado, pico_idx_ref = janela_adaptativa(audio_ref, fs_ref)

    linhas.append("=== Sinal de referência (AT9_impulse_03) ===")
    linhas.append(f"{'Banda(Hz)':<12}{'D50(%)':<12}{'C50(dB)':<10}")
    for fc in f_central_lista:
        C50, D50 = calcular_c50_d50(audio_cortado, fs_ref, fc, pico_idx=pico_idx_ref)
        linhas.append(f"{fc:<12}{D50*100:<12.3f}{C50:<10.3f}")
    linhas.append("")

    # ----- Por sala -----
    C50_por_sala = {sala_id: [] for sala_id, _ in SALAS}
    D50_por_sala = {sala_id: [] for sala_id, _ in SALAS}

    for sala_id, exclude_tok in SALAS:
        files = _coletar_arquivos(sala_id, exclude_tok)
        linhas.append(f"=== Sala {sala_id} ===")
        linhas.append(
            f"{'Arquivo':<22}{'Banda(Hz)':<12}{'D50(%)':<12}{'C50(dB)':<10}"
        )
        if not files:
            linhas.append("(nenhum arquivo encontrado)")
            linhas.append("")
            continue

        for filepath in files:
            fs, audio = wav.read(filepath)
            audio_norm = audio / np.max(np.abs(audio))
            audio_cortado, pico_idx = janela_adaptativa(audio_norm, fs)
            fname = os.path.splitext(os.path.basename(filepath))[0]

            print(f"Processando {fname}... ", end='', flush=True)
            c50_linha = []
            d50_linha = []
            for fc in f_central_lista:
                C50, D50 = calcular_c50_d50(audio_cortado, fs, fc, pico_idx=pico_idx)
                c50_linha.append(C50)
                d50_linha.append(D50 * 100)
                if np.isnan(C50) or np.isnan(D50):
                    linhas.append(
                        f"{fname:<22}{fc:<12}{'NaN':<12}{'NaN':<10}"
                    )
                else:
                    linhas.append(
                        f"{fname:<22}{fc:<12}{D50*100:<12.3f}{C50:<10.3f}"
                    )
            C50_por_sala[sala_id].append(c50_linha)
            D50_por_sala[sala_id].append(d50_linha)
            print("ok")
        linhas.append("")

    # ----- Médias por sala -----
    header_medias = (
        f"{'Sala':<6}{'N':<5}"
        + "".join([f"{int(fc):<10}" for fc in f_central_lista])
    )

    linhas.append("=== Médias por sala — C50 (dB) ===")
    linhas.append(header_medias)
    for sala_id, _ in SALAS:
        valores = C50_por_sala[sala_id]
        if not valores:
            linhas.append(f"{sala_id:<6}0")
            continue
        media = np.nanmean(np.array(valores), axis=0)
        row = f"{sala_id:<6}{len(valores):<5}"
        row += "".join([f"{v:<10.3f}" for v in media])
        linhas.append(row)
    linhas.append("")

    linhas.append("=== Médias por sala — D50 (%) ===")
    linhas.append(header_medias)
    for sala_id, _ in SALAS:
        valores = D50_por_sala[sala_id]
        if not valores:
            linhas.append(f"{sala_id:<6}0")
            continue
        media = np.nanmean(np.array(valores), axis=0)
        row = f"{sala_id:<6}{len(valores):<5}"
        row += "".join([f"{v:<10.3f}" for v in media])
        linhas.append(row)

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    print(f"\nResumo escrito em {OUT_FILE}")

# %%
