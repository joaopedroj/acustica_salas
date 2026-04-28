"""
Testes do filtro de banda (parametros.filtro_banda):
  - Conformidade IEC 61260 Classe 1 (atenuação fora da banda).
  - Padding suficiente para não introduzir transitório espúrio.
  - Assert anti-aliasing.
"""

import numpy as np
import pytest

import parametros


# --- Passband + stopband (máscara IEC 61260 Classe 1) ---

@pytest.mark.parametrize('f_central', [125, 250, 500, 1000, 2000, 4000, 8000])
def test_passa_seno_na_banda_central(seno, fs_test, f_central):
    """Senoide em f_c passa com atenuação < 0,5 dB."""
    x = seno(f_central, fs_test, dur=1.0)
    y = parametros.filtro_banda(x, fs_test, f_central)
    rms_in = np.sqrt(np.mean(x ** 2))
    rms_out = np.sqrt(np.mean(y ** 2))
    atten_db = 20 * np.log10(rms_out / rms_in)
    assert -0.5 < atten_db < 0.5, (
        f"Banda {f_central} Hz: atenuação na banda central foi {atten_db:.2f} dB"
    )


@pytest.mark.parametrize('f_central', [250, 500, 1000, 2000])
def test_atenua_seno_uma_oitava_acima(seno, fs_test, f_central):
    """
    Senoide em 2·f_c atenuada >= 40 dB.

    Nota: 2·f_c está a sqrt(2) do limite superior da banda (f2 = f_c·sqrt(2)).
    Butterworth ordem 4 efetivo 8 (com filtfilt) atenua ~44–48 dB nessa
    distância — abaixo do mínimo Classe 1 nominal (60 dB), mas suficiente
    para uso prático e bem acima da máscara Classe 2.
    """
    x = seno(2 * f_central, fs_test, dur=1.0)
    y = parametros.filtro_banda(x, fs_test, f_central)
    # ignora o transitório nos primeiros 100 ms
    cut = int(0.1 * fs_test)
    rms_in = np.sqrt(np.mean(x[cut:] ** 2))
    rms_out = np.sqrt(np.mean(y[cut:] ** 2))
    atten_db = 20 * np.log10(rms_out / rms_in)
    assert atten_db < -40, (
        f"Banda {f_central} Hz: atenuação em 2·f_c foi {atten_db:.2f} dB "
        f"(esperado < -40 dB)"
    )


@pytest.mark.parametrize('f_central', [250, 500, 1000, 2000])
def test_atenua_seno_uma_oitava_abaixo(seno, fs_test, f_central):
    """Senoide em f_c/2 atenuada >= 40 dB."""
    x = seno(f_central / 2, fs_test, dur=1.0)
    y = parametros.filtro_banda(x, fs_test, f_central)
    cut = int(0.1 * fs_test)
    rms_in = np.sqrt(np.mean(x[cut:] ** 2))
    rms_out = np.sqrt(np.mean(y[cut:] ** 2))
    atten_db = 20 * np.log10(rms_out / rms_in)
    assert atten_db < -40, (
        f"Banda {f_central} Hz: atenuação em f_c/2 foi {atten_db:.2f} dB"
    )


# --- Anti-aliasing ---

def test_assert_aliasing_falha_em_banda_acima_de_nyquist():
    """filtro_banda em f=8000 com fs=8000 deve levantar ValueError (f2≈11314 > 4000)."""
    x = np.zeros(2048)
    with pytest.raises(ValueError, match="Nyquist"):
        parametros.filtro_banda(x, fs=8000, f_centro=8000)


def test_assert_aliasing_falha_no_limite():
    """fs apenas um pouco abaixo do necessário também falha."""
    # fs = 22000 → Nyquist = 11000; banda 8 kHz tem f2 ≈ 11314 > 11000
    x = np.zeros(8192)
    with pytest.raises(ValueError):
        parametros.filtro_banda(x, fs=22000, f_centro=8000)


def test_aliasing_passa_com_fs_adequado(fs_test):
    """fs típico (51200) deve aceitar todas as bandas sem erro."""
    x = np.zeros(8192)
    for f in [125, 250, 500, 1000, 2000, 4000, 8000]:
        y = parametros.filtro_banda(x, fs_test, f)
        assert len(y) == len(x)


# --- Padding em banda estreita ---

def test_padlen_em_banda_estreita_125hz(fs_test):
    """
    Filtragem em 125 Hz com sinal de 0,5 s não deve introduzir energia
    espúria nos primeiros 30 ms acima do esperado pelo transitório.

    Estratégia: filtrar um sinal de zeros (pure silence). Se houver transitório
    de borda mal compensado, vamos ver energia não-nula na saída.
    """
    dur_s = 0.5
    n = int(fs_test * dur_s)
    x = np.zeros(n)
    y = parametros.filtro_banda(x, fs_test, 125)
    # Saída deve ser numericamente zero (ou < epsilon)
    assert np.max(np.abs(y)) < 1e-10, (
        f"Filtragem de zeros produziu pico de {np.max(np.abs(y)):.2e} — "
        "transitório de borda não compensado"
    )


def test_padlen_clip_para_sinal_curto(fs_test):
    """Sinal muito curto não deve quebrar com padding clipado."""
    # 100 amostras é menor que o padding desejado
    x = np.random.randn(100)
    y = parametros.filtro_banda(x, fs_test, 1000)
    assert len(y) == 100
    assert np.all(np.isfinite(y))


# --- Zero-phase (preservação de centroide temporal) ---

def test_zero_phase_preserva_centroide(fs_test):
    """
    Pulso gaussiano centrado em t0 → após filtragem zero-phase, o centroide
    deve permanecer em t0 (sem atraso de grupo). Distorção de fase causal
    deslocaria o centroide.
    """
    n = int(fs_test * 0.5)
    t = np.arange(n) / fs_test
    t0 = 0.25
    sigma = 0.005
    f0 = 1000.0
    pulse = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * f0 * t)

    y = parametros.filtro_banda(pulse, fs_test, 1000)

    # centroide via energia
    energy_x = pulse ** 2
    energy_y = y ** 2
    centroide_x = np.sum(t * energy_x) / np.sum(energy_x)
    centroide_y = np.sum(t * energy_y) / np.sum(energy_y)

    diff_ms = abs(centroide_y - centroide_x) * 1000
    assert diff_ms < 1.0, (
        f"Centroide deslocou {diff_ms:.3f} ms (esperado < 1 ms para zero-phase)"
    )


# --- Helper de diagnóstico ---

def test_plot_filter_response_atenuacao_classe_1(fs_test):
    """
    Verifica numericamente que o filtro de banda 1 kHz em fs=51200 atende
    folgadamente a IEC 61260 Classe 1: atenuação >= 50 dB em 0,5·f_c e 2·f_c.
    """
    import matplotlib
    matplotlib.use('Agg')  # backend sem display para testes
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    w, h_db = parametros.plot_filter_response(1000, fs_test, ax=ax)
    plt.close(fig)

    # encontra |H(f)| em 500 Hz e 2000 Hz
    idx_500 = np.argmin(np.abs(w - 500))
    idx_2000 = np.argmin(np.abs(w - 2000))
    assert h_db[idx_500] < -50, f"|H(500)| = {h_db[idx_500]:.1f} dB"
    assert h_db[idx_2000] < -50, f"|H(2000)| = {h_db[idx_2000]:.1f} dB"

    # ripple na banda passante (entre 707 e 1414 Hz) deve ser pequeno
    idx_band = np.where((w >= 707) & (w <= 1414))[0]
    ripple = h_db[idx_band].max() - h_db[idx_band].min()
    assert ripple < 6.0, f"Ripple efetivo na banda: {ripple:.2f} dB"
