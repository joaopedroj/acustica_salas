#%%

import os
import sys
import glob
import numpy as np
import scipy.io.wavfile as wav

# parametros.py em scripts/
_PARENT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
# NPS.py em scripts/nps/ (irmão)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# claridade.py em scripts/claridade/
_CLARIDADE = os.path.join(_PARENT, 'claridade')
if _CLARIDADE not in sys.path:
    sys.path.insert(0, _CLARIDADE)

import parametros
from NPS import UTR, calcular_snr
from claridade import janela_adaptativa, calcular_c50_d50

AUDIO_DIR = os.path.join(_HERE, '..', '..', 'audio')
OUT_DIR = os.path.join(_HERE, 'resultados')
OUT_FILE = os.path.join(OUT_DIR, 'NPS_resumo.txt')
P_REF = 20e-6

# (sala, pink_sv, bg_sv, pink_cv, vent_cv, impulsos_alvo)
# impulsos_alvo: lista explícita de basenames (sem .wav).
CONFIGS = [
    ('AT5',
     'AT5_noise_pink_mic1.wav', 'AT5_noise_background_mic1.wav',
     'AT5_noise_pink_ventilador_mic1.wav', 'AT5_noise_ventilator_mic1.wav',
     [f'AT5_impulse_{i:02d}' for i in range(1, 9)]),
    ('AT9',
     'AT9_noise_pink_1.wav', 'AT9_noise_background_1.wav',
     'AT9_noise_pink_ventilador1.wav', 'AT9_noise_ventilator_1.wav',
     [f'AT9_impulse_{i:02d}' for i in range(1, 9)]),
    ('AT10',
     'AT10_noise_pink_1.wav', 'AT10_noise_background_1.wav',
     'AT10_noise_pink_ventilador.wav', 'AT10_noise_ventilator_1.wav',
     [f'AT10_impulse_{i:02d}' for i in range(1, 9)]),
]


def _coletar_impulsos(alvo):
    files = []
    for nome in alvo:
        p = os.path.join(AUDIO_DIR, f'{nome}.wav')
        if os.path.exists(p):
            files.append(p)
        else:
            print(f"AVISO: arquivo não encontrado: {p}")
    return files


def _u50_por_arquivo(files, snr_por_banda, fs_target=None):
    """Para cada arquivo, devolve lista de U50 por banda (mesma ordem da lista de bandas)."""
    f_central_lista = parametros.f_central_lista
    resultado = []  # lista de (fname, [U50_por_banda])
    for fp in files:
        fname = os.path.splitext(os.path.basename(fp))[0]
        fs, audio = wav.read(fp)
        audio_norm = audio / np.max(np.abs(audio))
        audio_cortado, pico_idx = janela_adaptativa(audio_norm, fs)
        u50_linha = []
        for fc, snr_b in zip(f_central_lista, snr_por_banda):
            C50, _ = calcular_c50_d50(audio_cortado, fs, fc, pico_idx=pico_idx)
            if np.isnan(C50):
                u50_linha.append(float('nan'))
            else:
                u50_linha.append(UTR(C50, snr_b))
        resultado.append((fname, u50_linha))
    return resultado


def _escrever_bloco_por_arquivo(linhas, titulo, resultados, f_central_lista):
    linhas.append(f"=== {titulo} ===")
    linhas.append(f"{'Arquivo':<22}{'Banda(Hz)':<12}{'U50(dB)':<10}")
    for fname, u50_linha in resultados:
        for fc, u in zip(f_central_lista, u50_linha):
            if np.isnan(u):
                linhas.append(f"{fname:<22}{fc:<12}{'NaN':<10}")
            else:
                linhas.append(f"{fname:<22}{fc:<12}{u:<10.3f}")
    linhas.append("")


def _escrever_tabela_medias(linhas, titulo, por_sala_u50, f_central_lista):
    linhas.append(f"=== {titulo} ===")
    linhas.append(f"{'Sala':<6}{'N':<5}"
                  + "".join([f"{int(fc):<10}" for fc in f_central_lista]))
    for sala_id, valores in por_sala_u50.items():
        if not valores:
            linhas.append(f"{sala_id:<6}0")
            continue
        arr = np.array(valores)
        media = np.nanmean(arr, axis=0)
        row = f"{sala_id:<6}{len(valores):<5}"
        row += "".join([f"{v:<10.3f}" for v in media])
        linhas.append(row)
    linhas.append("")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    f_central_lista = parametros.f_central_lista

    linhas = []

    # 1) SNR por sala (sem/com vent)
    snrs = {}
    linhas.append("=== SNR por sala (dB) ===")
    linhas.append(f"{'Sala':<6}{'Config':<12}"
                  + "".join([f"{int(fc):<10}" for fc in f_central_lista]))
    for sala, pink_sv, bg_sv, pink_cv, vent_cv, _ in CONFIGS:
        print(f"\n[{sala}] SNR sem vent:")
        snr_sv, *_x = calcular_snr(os.path.join(AUDIO_DIR, pink_sv),
                                    os.path.join(AUDIO_DIR, bg_sv),
                                    f_central_lista, P_REF)
        print(f"[{sala}] SNR com vent:")
        snr_cv, *_x = calcular_snr(os.path.join(AUDIO_DIR, pink_cv),
                                    os.path.join(AUDIO_DIR, vent_cv),
                                    f_central_lista, P_REF)
        snrs[sala] = {'sv': snr_sv, 'cv': snr_cv}
        linhas.append(f"{sala:<6}{'sem vent':<12}"
                      + "".join([f"{v:<10.3f}" for v in snr_sv]))
        linhas.append(f"{sala:<6}{'com vent':<12}"
                      + "".join([f"{v:<10.3f}" for v in snr_cv]))
    linhas.append("")

    # 2) U50 por arquivo (acumulando por sala)
    u50_sv_por_sala = {sala: [] for sala, *_ in CONFIGS}
    u50_cv_por_sala = {sala: [] for sala, *_ in CONFIGS}
    resultados_sv = []   # [(fname, [u50_banda]), ...] (todas as salas concatenadas)
    resultados_cv = []

    for sala, _, _, _, _, alvo in CONFIGS:
        files = _coletar_impulsos(alvo)
        print(f"\n[{sala}] computando U50 para {len(files)} impulsos...")
        res_sv = _u50_por_arquivo(files, snrs[sala]['sv'])
        res_cv = _u50_por_arquivo(files, snrs[sala]['cv'])
        resultados_sv.extend(res_sv)
        resultados_cv.extend(res_cv)
        u50_sv_por_sala[sala].extend([linha for _, linha in res_sv])
        u50_cv_por_sala[sala].extend([linha for _, linha in res_cv])

    _escrever_bloco_por_arquivo(
        linhas, "U50 sem ventilador (dB) — por arquivo",
        resultados_sv, f_central_lista,
    )
    _escrever_bloco_por_arquivo(
        linhas, "U50 com ventilador (dB) — por arquivo",
        resultados_cv, f_central_lista,
    )

    _escrever_tabela_medias(
        linhas, "Médias por sala — U50 sem vent (dB)",
        u50_sv_por_sala, f_central_lista,
    )
    _escrever_tabela_medias(
        linhas, "Médias por sala — U50 com vent (dB)",
        u50_cv_por_sala, f_central_lista,
    )

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    print(f"\nResumo escrito em {OUT_FILE}")

# %%
