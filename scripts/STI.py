"""
STI.py — Cálculo do Índice de Transmissão da Fala (Speech Transmission Index)
conforme IEC 60268-16, em 7 bandas de oitava × 14 frequências de modulação.
"""
#%%
import numpy as np
from scipy.signal import convolve

import parametros


# Frequências de modulação (IEC 60268-16)
F_MOD = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

DUR = 5.0      # duração de cada sinal (s)
M_X = 1.0      # profundidade de modulação


def gerar_sinais_anecoicos(fs, f_central_lista, dur=DUR, seed=0):
    """
    Gera, para cada banda, um sinal de ruído branco filtrado.
    Semente fixa (seed=0 por default) garante reprodutibilidade.
    """
    rng = np.random.default_rng(seed=seed)
    t_sig = np.linspace(0, dur, int(fs * dur), endpoint=False)
    noise = rng.standard_normal(len(t_sig))

    sinais = []
    for fc in f_central_lista:
        xk = parametros.filtro_banda(noise, fs, fc)
        sinais.append(xk)
    return t_sig, sinais


def calcular_mti(h_t, fs, f_central_lista, t_sig, sinais_anecoicos,
                 f_mod=F_MOD, m_x=M_X, fc_envelope=100.0):
    """
    Calcula a matriz TI (bandas × freq_mod) e retorna o vetor MTI por banda.

    Args:
        h_t: resposta ao impulso (já recortada).
        fs: frequência de amostragem.
        f_central_lista: lista de bandas de oitava.
        t_sig, sinais_anecoicos: saída de gerar_sinais_anecoicos().
        f_mod: frequências de modulação (default IEC 60268-16).
        m_x: profundidade de modulação (default 1.0).
        fc_envelope: frequência de corte do passa-baixa do envelope (default 100 Hz).

    Returns:
        MTI: array de tamanho len(f_central_lista) com a média por banda.
    """
    TI = np.zeros((len(f_central_lista), len(f_mod)))
    for k, xk in enumerate(sinais_anecoicos):
        for m, fm in enumerate(f_mod):
            mod = np.sqrt(1 + m_x * np.cos(2 * np.pi * fm * t_sig))
            xkm = xk * mod
            ykm = convolve(xkm, h_t, mode='full')
            if len(ykm) > len(xkm):
                ykm_analysis = ykm[:len(xkm)]
            else:
                ykm_analysis = np.pad(ykm, (0, len(xkm) - len(ykm)), 'constant')

            I_kmx = parametros.calcular_intensidade_instantanea(xkm)
            I_kmy = parametros.calcular_intensidade_instantanea(ykm_analysis)

            I_kmx_env = parametros.filtro_passa_baixa(I_kmx, fc_envelope, fs)
            I_kmy_env = parametros.filtro_passa_baixa(I_kmy, fc_envelope, fs)

            m_xkm = parametros.calcular_amplitude_modulacao_correlacao(
                I_kmx_env, fm, fs)
            m_ykm = parametros.calcular_amplitude_modulacao_correlacao(
                I_kmy_env, fm, fs)

            if m_xkm > m_ykm and m_xkm > 1e-6 and m_ykm >= 0:
                denom = m_xkm - m_ykm
                if denom > 1e-6:
                    SNR_ef = 10 * np.log10(m_xkm / denom)
                    TI[k, m] = np.clip((SNR_ef + 15.0) / 30.0, 0.0, 1.0)
                else:
                    TI[k, m] = 0.0
            else:
                TI[k, m] = 0.0
    return np.mean(TI, axis=1)


def calcular_sti_final(MTI):
    """Combina MTI por banda no STI final usando os pesos α_k e β_k."""
    alpha_k = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta_k = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
    term1 = np.sum(alpha_k * MTI)
    term2 = np.sum(beta_k * np.sqrt(MTI[1:] * MTI[1:]))
    return term1 - term2, term1, term2


if __name__ == "__main__":
    _default = parametros.load_default_audio()
    h_audio = _default['audio']
    fs = _default['fs']
    f_central_lista = parametros.f_central_lista

    max_pos = int(np.argmax(h_audio))
    soma_indice = 28230
    h_t = h_audio[max_pos - 600: max_pos + soma_indice]

    t_sig, sinais_anecoicos = gerar_sinais_anecoicos(fs, f_central_lista)

    MTI = calcular_mti(h_t, fs, f_central_lista, t_sig, sinais_anecoicos)
    print(f"\nMTI por banda: {MTI}")

    STI, term1, term2 = calcular_sti_final(MTI)

    print(f"\nResultados finais:")
    print(f"Termo 1 (peso de banda central): {term1:.4f}")
    print(f"Termo 2 (peso de bandas adjacentes): {term2:.4f}")
    print(f"STI = {STI:.4f}")

    if STI >= 0.75:
        interpretacao = "Excelente"
    elif STI >= 0.60:
        interpretacao = "Boa"
    elif STI >= 0.45:
        interpretacao = "Regular"
    elif STI >= 0.30:
        interpretacao = "Pobre"
    else:
        interpretacao = "Muito pobre"

    print(f"Qualidade: {interpretacao}")

# %%
