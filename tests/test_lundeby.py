"""
Testes do método de Lundeby (parametros.lundeby_truncation_point):
  - Decaimento puro: ponto de truncamento próximo ao fim do sinal.
  - Decaimento com piso de ruído: detecta o cruzamento.
  - Casos limites (sinal curto, baixo SNR): retorna len(energy) sem crashar.
"""

import numpy as np
import pytest

import parametros


def test_decaimento_puro_sem_ruido(fs_test):
    """
    Exponencial sem ruído → ponto de truncamento próximo do fim do sinal.
    """
    dur = 2.0
    n = int(fs_test * dur)
    t = np.arange(n) / fs_test
    T60 = 0.7
    energy = np.exp(-13.816 * t / T60)  # exp em energia (2x slope da amplitude)

    idx = parametros.lundeby_truncation_point(energy, fs_test)

    # Sem piso, o algoritmo pode não convergir e retornar len(energy),
    # ou convergir muito ao fim. Aceita-se >= 80% do sinal.
    assert idx >= int(0.8 * n), (
        f"Sem ruído, esperava-se truncamento próximo ao fim; obtido idx={idx} "
        f"de {n}"
    )


def test_decaimento_com_piso_de_ruido(fs_test):
    """
    Exponencial com piso de ruído branco a -50 dB:
    o ponto de truncamento deve cair perto do tempo de cruzamento teórico
    com a curva de decaimento.
    """
    dur = 2.0
    n = int(fs_test * dur)
    t = np.arange(n) / fs_test
    T60 = 0.5  # decaimento em amplitude → energia cai 2x mais rápido

    # energia exponencial em amplitude → squared decay
    amp_decay = np.exp(-6.908 * t / T60)
    rng = np.random.default_rng(0)
    noise = 1e-3 * rng.standard_normal(n)  # ruído de amplitude
    signal = amp_decay + noise
    energy = signal ** 2

    idx = parametros.lundeby_truncation_point(energy, fs_test)
    t_cross = idx / fs_test

    # Cruzamento teórico: amp_decay = noise_rms ≈ 1e-3
    # → t_cross_teorico = T60 * log(1/1e-3) / 6.908 ≈ 0.5
    assert 0.3 < t_cross < 0.9, (
        f"Cruzamento em {t_cross:.3f} s; esperado ~0,5 s ± 0,3 s"
    )


def test_baixo_snr_nao_crasha(fs_test):
    """SNR muito baixo: função deve retornar len(energy), sem exceção."""
    n = int(fs_test * 1.5)
    rng = np.random.default_rng(0)
    energy = (1e-3 * rng.standard_normal(n)) ** 2  # só ruído

    idx = parametros.lundeby_truncation_point(energy, fs_test)
    assert isinstance(idx, int)
    assert 0 < idx <= n


def test_sinal_muito_curto_retorna_n(fs_test):
    """Sinal < 0,5 s: função retorna len(energy) sem tentar Lundeby."""
    n = int(fs_test * 0.2)  # 200 ms
    energy = np.random.rand(n)
    idx = parametros.lundeby_truncation_point(energy, fs_test)
    assert idx == n


def test_energia_zero_nao_crasha(fs_test):
    """Sinal todo zero → retorna len(energy)."""
    n = int(fs_test * 1.0)
    energy = np.zeros(n)
    idx = parametros.lundeby_truncation_point(energy, fs_test)
    assert idx == n


def test_idx_no_intervalo_valido(fs_test):
    """O índice retornado deve estar em [1, len(energy)] sempre."""
    n = int(fs_test * 1.0)
    rng = np.random.default_rng(42)
    energy = np.exp(-2 * np.arange(n) / fs_test) + 1e-2 * rng.random(n)
    idx = parametros.lundeby_truncation_point(energy, fs_test)
    assert 1 <= idx <= n
