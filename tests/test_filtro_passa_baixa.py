"""
Testes do filtro passa-baixa (parametros.filtro_passa_baixa):
  - O parâmetro fc deve ser respeitado (sem cap interno).
  - Filtragem zero-phase via sosfiltfilt.
  - Atenuação adequada em f >= 2·fc.
"""

import numpy as np
import pytest

import parametros


def test_fc_respeitado(seno, fs_test):
    """fc=200 Hz: senoide em 150 Hz passa quase intacta (atenuação > -3 dB)."""
    x = seno(150, fs_test, dur=1.0)
    y = parametros.filtro_passa_baixa(x, fc=200.0, fs_signal=fs_test)
    cut = int(0.1 * fs_test)
    rms_in = np.sqrt(np.mean(x[cut:] ** 2))
    rms_out = np.sqrt(np.mean(y[cut:] ** 2))
    atten_db = 20 * np.log10(rms_out / rms_in)
    assert atten_db > -3, (
        f"150 Hz com fc=200 Hz teve atenuação {atten_db:.2f} dB (esperado > -3)"
    )


def test_atenua_acima_da_corte(seno, fs_test):
    """Senoide em 2·fc atenuada >= 25 dB (Butterworth 4 + sosfiltfilt = 8 efetivo)."""
    fc = 100.0
    x = seno(200, fs_test, dur=1.0)
    y = parametros.filtro_passa_baixa(x, fc=fc, fs_signal=fs_test)
    cut = int(0.1 * fs_test)
    rms_in = np.sqrt(np.mean(x[cut:] ** 2))
    rms_out = np.sqrt(np.mean(y[cut:] ** 2))
    atten_db = 20 * np.log10(rms_out / rms_in)
    assert atten_db < -25, (
        f"f=200 Hz com fc=100 Hz teve atenuação {atten_db:.2f} dB (esperado < -25)"
    )


def test_passa_abaixo_da_corte(seno, fs_test):
    """Senoide em 0,5·fc passa quase intacta."""
    fc = 100.0
    x = seno(50, fs_test, dur=1.0)
    y = parametros.filtro_passa_baixa(x, fc=fc, fs_signal=fs_test)
    cut = int(0.1 * fs_test)
    rms_in = np.sqrt(np.mean(x[cut:] ** 2))
    rms_out = np.sqrt(np.mean(y[cut:] ** 2))
    atten_db = 20 * np.log10(rms_out / rms_in)
    assert atten_db > -1, f"50 Hz com fc=100 Hz teve atenuação {atten_db:.2f} dB"


def test_zero_phase_pico_nao_atrasa(fs_test):
    """Pulso filtrado em modo zero-phase mantém o pico no mesmo índice."""
    n = int(fs_test * 0.5)
    pulse = np.zeros(n)
    pulse[n // 2] = 1.0
    y = parametros.filtro_passa_baixa(pulse, fc=500.0, fs_signal=fs_test)
    pico_y = int(np.argmax(np.abs(y)))
    pico_x = n // 2
    diff = abs(pico_y - pico_x)
    diff_ms = diff / fs_test * 1000
    assert diff_ms < 0.5, (
        f"Pico deslocou {diff} amostras ({diff_ms:.3f} ms); esperado < 0,5 ms"
    )


def test_fc_acima_de_nyquist_retorna_inalterado(fs_test):
    """Se fc >= Nyquist, função retorna o sinal sem filtrar (sem crash)."""
    x = np.random.randn(1024)
    y = parametros.filtro_passa_baixa(x, fc=fs_test, fs_signal=fs_test)
    np.testing.assert_array_equal(x, y)


def test_sinal_curto_nao_quebra(fs_test):
    """Padding clipado para sinais curtos (regressão de robustez)."""
    x = np.random.randn(50)
    y = parametros.filtro_passa_baixa(x, fc=100.0, fs_signal=fs_test)
    assert len(y) == 50
    assert np.all(np.isfinite(y))
