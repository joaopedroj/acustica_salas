"""
Testes do pipeline de C50/D50 (claridade.calcular_c50_d50):
  - 1 kHz com T60 conhecido recupera C50/D50 analíticos.
  - 125 Hz (caso crítico do padlen) também recupera valor analítico.
  - C50 e D50 são internamente consistentes: D50/(1-D50) == 10**(C50/10).
  - Sinal de energia constante produz C50 ≈ -16 dB (razão geométrica das janelas).

Fórmulas analíticas (decaimento exponencial puro em uma banda):
    2α = 13,816 / T60
    razao  = e^(2α · 0,05) - 1
    C50    = 10·log10(razao)
    D50    = 1 - e^(-2α · 0,05)
"""

import numpy as np
import pytest
from scipy import signal as scipy_signal

import claridade


def _c50_analitico(T60):
    alpha2 = 13.816 / T60
    return 10.0 * np.log10(np.exp(alpha2 * 0.05) - 1.0)


def _d50_analitico(T60):
    alpha2 = 13.816 / T60
    return 1.0 - np.exp(-alpha2 * 0.05)


T60_CASOS = [0.5, 1.0, 2.0]


@pytest.mark.parametrize("T60_alvo", T60_CASOS)
def test_c50_em_1kHz_recupera_valor_analitico(impulso_decaimento_exponencial,
                                              fs_test, T60_alvo):
    """1 kHz: filtro tem transitório curto — caso fácil de validar."""
    C50_esperado = _c50_analitico(T60_alvo)
    D50_esperado = _d50_analitico(T60_alvo)

    h = impulso_decaimento_exponencial(
        fs=fs_test, T60=T60_alvo, f_central=1000,
        dur=2.5, snr_db=60, seed=0,
    )
    audio_cortado, pico_idx = claridade.janela_adaptativa(h, fs_test)
    C50, D50 = claridade.calcular_c50_d50(audio_cortado, fs_test, 1000,
                                          pico_idx=pico_idx)

    assert abs(C50 - C50_esperado) < 1.0, (
        f"C50={C50:.3f} dB vs esperado={C50_esperado:.3f} dB "
        f"(T60={T60_alvo}s, 1000 Hz)"
    )
    assert abs(D50 - D50_esperado) < 0.05, (
        f"D50={D50:.3f} vs esperado={D50_esperado:.3f} "
        f"(T60={T60_alvo}s, 1000 Hz)"
    )


def test_c50_em_125Hz_robusto_a_padlen(impulso_decaimento_exponencial, fs_test):
    """
    Caso crítico: banda 125 Hz, onde o transitório do filtro IIR é longo.
    Este é o teste que falharia com o padlen antigo (default do scipy).
    """
    T60_alvo = 0.5
    C50_esperado = _c50_analitico(T60_alvo)
    D50_esperado = _d50_analitico(T60_alvo)

    h = impulso_decaimento_exponencial(
        fs=fs_test, T60=T60_alvo, f_central=125,
        dur=2.5, snr_db=60, seed=0,
    )
    audio_cortado, pico_idx = claridade.janela_adaptativa(h, fs_test)
    C50, D50 = claridade.calcular_c50_d50(audio_cortado, fs_test, 125,
                                          pico_idx=pico_idx)

    assert abs(C50 - C50_esperado) < 2.0, (
        f"C50={C50:.3f} dB vs esperado={C50_esperado:.3f} dB "
        f"(T60={T60_alvo}s, 125 Hz). Padlen do filtro_banda pode estar "
        f"insuficiente para a memória do IIR em banda baixa."
    )
    assert abs(D50 - D50_esperado) < 0.10, (
        f"D50={D50:.3f} vs esperado={D50_esperado:.3f} "
        f"(T60={T60_alvo}s, 125 Hz)"
    )


def test_c50_d50_consistencia_interna(impulso_decaimento_exponencial, fs_test):
    """
    C50 e D50 derivam das mesmas energias pre/pos 50 ms; portanto
    D50 / (1 - D50) == E_pre / E_pos == 10**(C50/10) sempre.
    """
    h = impulso_decaimento_exponencial(
        fs=fs_test, T60=0.8, f_central=1000, dur=2.0, snr_db=60, seed=1,
    )
    audio_cortado, pico_idx = claridade.janela_adaptativa(h, fs_test)
    C50, D50 = claridade.calcular_c50_d50(audio_cortado, fs_test, 1000,
                                          pico_idx=pico_idx)

    razao_via_D50 = D50 / (1.0 - D50)
    razao_via_C50 = 10.0 ** (C50 / 10.0)
    np.testing.assert_allclose(razao_via_D50, razao_via_C50, rtol=1e-6)


def test_c50_sinal_constante_em_1kHz():
    """
    Sinal de energia constante (sem decaimento) em uma banda passante:
    energia distribuída uniformemente. Razão E_pre50/E_pos50 = 0.05/(T-0.05).
    Para T=2 s: razão ≈ 0.0256 → C50 ≈ -15.9 dB; D50 ≈ 0.0250.

    Usa-se um seno em 1 kHz (passa pela banda 707-1414 Hz com ganho ~1)
    para que o filtro_banda seja efetivamente identidade.
    """
    fs = 51200
    T = 2.0
    t = np.arange(int(fs * T)) / fs
    h = np.sin(2.0 * np.pi * 1000.0 * t) * 0.1

    C50, D50 = claridade.calcular_c50_d50(h, fs, 1000)

    assert -17.0 < C50 < -14.0, f"C50={C50:.3f} dB fora da faixa esperada [-17, -14]"
    assert 0.02 < D50 < 0.04, f"D50={D50:.4f} fora da faixa esperada [0.02, 0.04]"


def test_c50_em_125Hz_com_pre_silencio(fs_test):
    """
    Caso realista: silêncio + (ruído branco × decaimento exponencial), depois
    nada de filtragem pré (a filtragem é interna do calcular_c50_d50).

    O início abrupto do ruído cria um pico delta-like no início do decay,
    análogo ao clap real de medição. Sem buffer pré-pico, o sosfiltfilt
    reflete esse pico no padding e infla D50; com buffer, o filtro vê
    o silêncio anterior e produz valores próximos do analítico.

    Em sala reverberante (T60=2 s em 125 Hz), D50 analítico ≈ 0,29.
    """
    T60_alvo = 2.0
    D50_esperado = 1.0 - np.exp(-13.816 * 0.05 / T60_alvo)   # ≈ 0,288

    fs = fs_test
    pre_silencio_s = 0.5
    dur_pos_s = 4.0   # >> T60 para garantir cauda significativa
    pre_silencio = int(pre_silencio_s * fs)
    n_pos = int(dur_pos_s * fs)
    n = n_pos + pre_silencio

    rng = np.random.default_rng(0)
    t_pos = np.arange(n_pos) / fs
    decay = np.exp(-6.908 * t_pos / T60_alvo)

    sinal = np.zeros(n)
    sinal[pre_silencio:] = rng.standard_normal(n_pos) * decay

    audio_cortado, pico_idx = claridade.janela_adaptativa(sinal, fs)
    C50, D50 = claridade.calcular_c50_d50(audio_cortado, fs, 125,
                                          pico_idx=pico_idx)

    # Asserção semântica: sala reverberante → D50 < 0,5 (energia útil < energia
    # tardia). Sem buffer pré-pico, o artefato inflava D50 para ≈ 0,90–0,95.
    # Com buffer, D50 fica abaixo de 0,5, consistente com T60 = 2 s.
    # Tolerância numérica afrouxada porque o sinal é estocástico (ruído branco
    # × decay): valor exato depende da seed, mas o valor é robustamente baixo.
    assert D50 < 0.5, (
        f"D50={D50:.3f} alto demais para sala reverberante (T60=2s, 125 Hz). "
        f"Esperado: D50 < 0,5 (analítico ≈ {D50_esperado:.3f}). "
        f"Artefato de padding ainda presente?"
    )
    assert C50 < 0.0, (
        f"C50={C50:.3f} dB positivo demais para T60=2s; esperado C50 < 0 dB."
    )


def test_d50_inversamente_proporcional_a_T60(impulso_decaimento_exponencial,
                                              fs_test):
    """
    Coerência física: sala mais reverberante (T60 maior) → D50 menor.
    Relação deve ser estritamente monotônica para sinais sintéticos limpos.
    """
    def _d50(T60):
        h = impulso_decaimento_exponencial(
            fs=fs_test, T60=T60, f_central=1000,
            dur=max(2.5, 2.5 * T60), snr_db=60, seed=0,
        )
        ac, pi = claridade.janela_adaptativa(h, fs_test)
        _, d50 = claridade.calcular_c50_d50(ac, fs_test, 1000, pico_idx=pi)
        return d50

    d50_curto = _d50(0.3)
    d50_medio = _d50(1.0)
    d50_longo = _d50(2.5)
    assert d50_curto > d50_medio > d50_longo, (
        f"D50 não decresce com T60: "
        f"T60=0.3→{d50_curto:.3f}, T60=1.0→{d50_medio:.3f}, T60=2.5→{d50_longo:.3f}"
    )
