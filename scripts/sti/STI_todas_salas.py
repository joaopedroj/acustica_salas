#%%

import os
import sys
import glob
import numpy as np
import scipy.io.wavfile as wav

# parametros.py está em scripts/ (pasta-pai)
_PARENT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# STI.py está nesta mesma pasta
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import parametros
from STI import gerar_sinais_anecoicos, calcular_mti, calcular_sti_final

AUDIO_DIR = os.path.join(_HERE, '..', '..', 'audio')
OUT_DIR = os.path.join(_HERE, 'resultados')
OUT_FILE = os.path.join(OUT_DIR, 'STI_todas_salas.txt')

# (id, exclude_token)
SALAS = [('AT5', None), ('AT9', None), ('AT10', 'nao_usado')]


def _qualidade(sti):
    if sti >= 0.75:
        return "Excelente"
    if sti >= 0.60:
        return "Boa"
    if sti >= 0.45:
        return "Regular"
    if sti >= 0.30:
        return "Pobre"
    return "Muito pobre"


def _recortar_h_t(audio, fs):
    """Mesma janela do STI.py original: [max-600 : max+28230] sobre o sinal normalizado."""
    audio_norm = audio / np.max(np.abs(audio))
    max_pos = int(np.argmax(np.abs(audio_norm)))
    inicio = max(max_pos - 600, 0)
    fim = min(max_pos + 28230, len(audio_norm))
    return audio_norm[inicio:fim]


def _coletar_arquivos(at_id, exclude_token):
    pattern = os.path.join(AUDIO_DIR, f'{at_id}_impulse_[0-9]*.wav')
    files = sorted(glob.glob(pattern))
    if exclude_token:
        files = [f for f in files if exclude_token not in f]
    return files


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    f_central_lista = parametros.f_central_lista

    # Sinais anecoicos determinísticos (seed=0). Cache por fs evita regerar.
    sinais_cache = {}

    linhas = []
    linhas.append("=== Resultados por arquivo ===")
    header = (
        f"{'Arquivo':<22}"
        + "".join([f"MTI_{int(fc):<6}" for fc in f_central_lista])
        + f"{'STI':<8}{'Qualidade':<12}"
    )
    linhas.append(header)

    stats = {sala_id: [] for sala_id, _ in SALAS}

    for sala_id, exclude_tok in SALAS:
        files = _coletar_arquivos(sala_id, exclude_tok)
        if not files:
            print(f"[{sala_id}] nenhum arquivo encontrado.")
            continue
        for fp in files:
            fname = os.path.splitext(os.path.basename(fp))[0]
            print(f"Processando {fname}... ", end='', flush=True)
            fs, raw = wav.read(fp)
            h_t = _recortar_h_t(raw, fs)

            if fs not in sinais_cache:
                sinais_cache[fs] = gerar_sinais_anecoicos(
                    fs, f_central_lista, seed=0
                )
            t_sig, sinais = sinais_cache[fs]

            MTI = calcular_mti(h_t, fs, f_central_lista, t_sig, sinais)
            STI_val, _, _ = calcular_sti_final(MTI)
            stats[sala_id].append(STI_val)

            row = f"{fname:<22}"
            row += "".join([f"{m:<10.3f}" for m in MTI])
            row += f"{STI_val:<8.3f}{_qualidade(STI_val):<12}"
            linhas.append(row)
            print(f"STI={STI_val:.3f}")

    linhas.append("")
    linhas.append("=== Médias por sala ===")
    linhas.append(
        f"{'Sala':<6}{'N_arquivos':<12}{'STI_médio':<12}"
        f"{'STI_dp':<10}{'Qualidade_média':<18}"
    )
    for sala_id, _ in SALAS:
        valores = np.array(stats[sala_id]) if stats[sala_id] else np.array([])
        if len(valores) == 0:
            linhas.append(f"{sala_id:<6}0")
            continue
        m = float(np.mean(valores))
        dp = float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0
        linhas.append(
            f"{sala_id:<6}{len(valores):<12}"
            f"{m:<12.3f}{dp:<10.3f}{_qualidade(m):<18}"
        )

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    print(f"\nResumo escrito em {OUT_FILE}")

# %%
