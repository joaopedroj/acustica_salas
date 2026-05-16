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

# T_reverb.py está nesta mesma pasta
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import parametros
from parametros import ajustar_t60
from T_reverb import computar_schroeder

AUDIO_DIR = os.path.join(_HERE, '..', '..', 'audio')
OUT_DIR = os.path.join(_HERE, 'resultados')
OUT_FILE = os.path.join(OUT_DIR, 'T60_resumo.txt')

# (id, exclude_token, noise_file, duracao_s)
CONFIGS = [
    ('AT5',  None,        'AT5_noise_background_mic1.wav', 0.8),
    ('AT9',  None,        'AT9_noise_background_1.wav',    1.8),
    ('AT10', 'nao_usado', 'AT10_noise_background_1.wav',   0.8),
]


def _coletar_arquivos(at_id, exclude_token):
    pattern = os.path.join(AUDIO_DIR, f'{at_id}_impulse_[0-9]*.wav')
    files = sorted(glob.glob(pattern))
    if exclude_token:
        files = [f for f in files if exclude_token not in f]
    return files


def _processar_sala(at_id, exclude_token, noise_file_name, duracao_s, lines):
    files = _coletar_arquivos(at_id, exclude_token)
    lines.append(
        f"=== Sala {at_id} (ruído de fundo: {noise_file_name}, "
        f"duração janela: {duracao_s}s) ==="
    )
    lines.append(
        f"{'Arquivo':<22}{'Banda(Hz)':<12}{'T20(s)':<10}"
        f"{'T30(s)':<10}{'T60(s)':<10}"
    )

    if not files:
        lines.append("(nenhum arquivo encontrado)")
        lines.append("")
        return

    for filepath in files:
        fname = os.path.splitext(os.path.basename(filepath))[0]
        print(f"Processando {fname}... ", end='', flush=True)
        fs, raw = wav.read(filepath)
        audio_pa = parametros.audio_pascal(raw, fs)
        audio_norm = raw / np.max(np.abs(raw))
        max_pos = int(np.argmax(np.abs(audio_norm)))
        soma = min(len(audio_pa) - max_pos, int(2.0 * duracao_s * fs))
        audio_cortado = audio_pa[max_pos: max_pos + soma]

        for fc in parametros.f_central_lista:
            _, sch_db, t_sch = computar_schroeder(audio_cortado, fs, fc)
            r = ajustar_t60(t_sch, sch_db)
            if r is None:
                lines.append(
                    f"{fname:<22}{fc:<12}{'NaN':<10}{'NaN':<10}{'NaN':<10}"
                )
                continue
            _, _, T20, T30, T60 = r
            lines.append(
                f"{fname:<22}{fc:<12}{T20:<10.3f}{T30:<10.3f}{T60:<10.3f}"
            )
        print("ok")
    lines.append("")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    for at_id, excl, noise_name, dur in CONFIGS:
        _processar_sala(at_id, excl, noise_name, dur, lines)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nResumo escrito em {OUT_FILE}")

# %%
