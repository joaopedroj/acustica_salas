"""
Testes do pipeline de T60 (T_reverb.computar_schroeder + parametros.ajustar_t60):
  - Janela ampla + Lundeby recupera T60 corretamente.
  - Integração com Lundeby para sinais com piso de ruído.
  - ajustar_t60 usa máscara [-5, -25] dB.
"""

import os
import sys
import numpy as np
import pytest

# Adiciona scripts/ e suas subpastas ao sys.path (caso não esteja via pytest.ini)
_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'scripts')
)
for _sub in ('', 'tempo_reverb', 'sti', 'claridade', 'nps'):
    _p = os.path.join(_SCRIPTS_DIR, _sub) if _sub else _SCRIPTS_DIR
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parametros
import T_reverb


def _gerar_h_t_sintetico(fs, T60_alvo, f_central, dur=2.5, snr_db=50, seed=0):
    """
    Resposta ao impulso sintética: ruído filtrado em banda + envelope
    exponencial com T60 alvo, mais piso de ruído branco.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs

    # ruído filtrado em banda via FFT
    carrier = rng.standard_normal(n)
    Y = np.fft.rfft(carrier)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    band_mask = (freqs >= f_central / np.sqrt(2)) & (freqs <= f_central * np.sqrt(2))
    Y[~band_mask] = 0
    carrier = np.fft.irfft(Y, n=n)
    carrier = carrier / np.max(np.abs(carrier))

    decay = np.exp(-6.908 * t / T60_alvo)
    h = carrier * decay

    peak = np.max(np.abs(h))
    noise_amp = peak * 10 ** (-snr_db / 20)
    h = h + noise_amp * rng.standard_normal(n)
    return h


def test_t60_recupera_decaimento_conhecido(fs_test):
    """
    Pipeline reproduz T60 alvo de 0,7 s em 1 kHz com erro < 10%.
    """
    T60_alvo = 0.7
    h = _gerar_h_t_sintetico(fs_test, T60_alvo, f_central=1000,
                             dur=2.5, snr_db=60, seed=0)
    _, schroeder_db, t_sch = T_reverb.computar_schroeder(h, fs_test, 1000)
    resultado = parametros.ajustar_t60(t_sch, schroeder_db)
    assert resultado is not None, "Falha ao ajustar T60"
    _, _, T20, T30, T60 = resultado

    erro_rel = abs(T60 - T60_alvo) / T60_alvo
    assert erro_rel < 0.10, (
        f"T60 recuperado {T60:.3f} s vs. alvo {T60_alvo:.3f} s (erro {erro_rel:.1%})"
    )


def test_t60_robusto_a_truncamento_pelo_lundeby(fs_test):
    """
    Sinal com piso de ruído alto (SNR=30 dB): Lundeby deve cortar a cauda
    contaminada e o T60 deve ser recuperado dentro de ±20%.
    """
    T60_alvo = 0.6
    h = _gerar_h_t_sintetico(fs_test, T60_alvo, f_central=1000,
                             dur=3.0, snr_db=30, seed=1)
    _, schroeder_db, t_sch = T_reverb.computar_schroeder(h, fs_test, 1000)
    resultado = parametros.ajustar_t60(t_sch, schroeder_db)
    assert resultado is not None
    _, _, _, _, T60 = resultado
    erro_rel = abs(T60 - T60_alvo) / T60_alvo
    assert erro_rel < 0.20, (
        f"T60={T60:.3f}s vs alvo={T60_alvo:.3f}s (erro {erro_rel:.1%}); "
        "Lundeby deveria ter compensado o piso de ruído"
    )


def test_ajustar_t60_usa_mask_minus5_minus25():
    """
    Verifica que ajustar_t60 ignora pontos acima de -5 dB e abaixo de -25 dB.
    Estratégia: criar curva onde os pontos fora dessa janela teriam slope
    diferente — o T60 deve refletir só o slope da janela [-5, -25].
    """
    fs = 51200
    n = int(fs * 1.0)
    t = np.arange(n) / fs

    # Curva sintética: -50 dB/s constante na faixa usada [-5, -25],
    # mas pontos fora têm slope diferente (corrompidos).
    schroeder_db = np.zeros(n)
    schroeder_db[t <= 0.05] = 0  # platô inicial (som direto)
    mask_linear = (t > 0.05) & (t < 0.6)
    schroeder_db[mask_linear] = -50 * (t[mask_linear] - 0.05)
    # Cauda corrompida (slope mais raso) — não deveria entrar no fit
    schroeder_db[t >= 0.6] = -27 - 5 * (t[t >= 0.6] - 0.6)

    resultado = parametros.ajustar_t60(t, schroeder_db)
    assert resultado is not None
    coeffs, _, _, _, T60 = resultado

    # Reta ajustada: y = -50·t + 2,5 (passa por (0.05, 0) com slope -50);
    # T60 satisfaz -60 = -50·T60 + 2,5 → T60 = 62,5/50 = 1,25 s.
    # (Slope -50 dB/s é o sinal de que a mask [-5,-25] foi respeitada;
    # se a mask incluísse o platô inicial ou a cauda corrompida, o slope
    # seria menos íngreme e T60 seria diferente.)
    T60_esperado = 1.25
    assert abs(T60 - T60_esperado) < 0.05, (
        f"T60={T60:.3f}s vs esperado={T60_esperado:.3f}s — "
        f"mask [-5,-25] não foi respeitada"
    )
    # Slope deve ser exatamente -50 dB/s (independente do intercept)
    assert abs(coeffs[0] - (-50.0)) < 0.5, (
        f"Slope ajustado = {coeffs[0]:.3f} dB/s; esperado -50 dB/s"
    )


def test_ajustar_t60_retorna_none_para_dados_insuficientes():
    """Curva que nunca atinge -5 dB → retorna None."""
    t = np.linspace(0, 1, 1000)
    schroeder_db = -t * 4  # max=-4 (nunca chega a -5)
    resultado = parametros.ajustar_t60(t, schroeder_db)
    assert resultado is None


def test_ajustar_t60_retorna_none_para_slope_positivo():
    """Slope positivo (sinal cresce) → retorna None (sentinela)."""
    t = np.linspace(0, 1, 1000)
    schroeder_db = -10 + 5 * t  # cresce de -10 para -5
    # mascara fica vazia ou degenerada
    resultado = parametros.ajustar_t60(t, schroeder_db)
    # Pode retornar None pela máscara vazia ou pelo slope >= 0
    assert resultado is None or resultado[0][0] < 0
