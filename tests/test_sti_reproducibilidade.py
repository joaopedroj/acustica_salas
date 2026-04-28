"""
Testes de reprodutibilidade do STI: gerador de ruído com semente fixa garante
que duas execuções consecutivas produzam o mesmo MTI/STI bit-a-bit.
"""

import os
import sys
import numpy as np

_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'scripts')
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import STI


def test_gerar_sinais_anecoicos_deterministico(fs_test):
    """Duas chamadas com mesma semente produzem arrays idênticos."""
    f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
    t1, sinais1 = STI.gerar_sinais_anecoicos(fs_test, f_central_lista,
                                              dur=1.0, seed=0)
    t2, sinais2 = STI.gerar_sinais_anecoicos(fs_test, f_central_lista,
                                              dur=1.0, seed=0)
    np.testing.assert_array_equal(t1, t2)
    for s1, s2 in zip(sinais1, sinais2):
        np.testing.assert_array_equal(s1, s2)


def test_gerar_sinais_anecoicos_seeds_diferentes(fs_test):
    """Sementes diferentes produzem arrays diferentes (sanidade)."""
    f_central_lista = [1000]
    _, sinais1 = STI.gerar_sinais_anecoicos(fs_test, f_central_lista,
                                              dur=1.0, seed=0)
    _, sinais2 = STI.gerar_sinais_anecoicos(fs_test, f_central_lista,
                                              dur=1.0, seed=1)
    assert not np.array_equal(sinais1[0], sinais2[0])


def test_calcular_mti_deterministico(fs_test):
    """
    O cálculo completo de MTI é determinístico dado h_t fixo e mesma seed.
    Smoke test rápido com h_t simples (exponencial em uma banda).
    """
    f_central_lista = [1000, 2000]   # subset para velocidade
    f_mod = [2.0, 4.0]               # subset para velocidade

    # h_t sintético: decaimento exponencial com T60 = 0,5 s, em banda
    n_h = int(fs_test * 0.6)
    t_h = np.arange(n_h) / fs_test
    rng = np.random.default_rng(7)
    h_t = rng.standard_normal(n_h) * np.exp(-13.816 * t_h / 0.5)

    t_sig, sinais = STI.gerar_sinais_anecoicos(fs_test, f_central_lista,
                                                dur=1.0, seed=0)

    MTI_a = STI.calcular_mti(h_t, fs_test, f_central_lista, t_sig, sinais,
                              f_mod=f_mod)
    MTI_b = STI.calcular_mti(h_t, fs_test, f_central_lista, t_sig, sinais,
                              f_mod=f_mod)
    np.testing.assert_array_equal(MTI_a, MTI_b)


def test_sti_final_funciona_com_mti_dummy():
    """calcular_sti_final retorna valor finito para MTI plausível."""
    MTI = np.array([0.4, 0.5, 0.6, 0.65, 0.7, 0.65, 0.5])
    STI_val, t1, t2 = STI.calcular_sti_final(MTI)
    assert np.isfinite(STI_val)
    assert -1 < STI_val < 1.5  # range plausível
