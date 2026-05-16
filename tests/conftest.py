"""
Fixtures compartilhadas para a suíte de testes.

`pythonpath = scripts` no pytest.ini garante que `import parametros` resolva
diretamente para acustica_salas/scripts/parametros.py.
"""

import os
import sys
import numpy as np
import pytest

# Garante import de parametros e dos scripts movidos para subpastas mesmo se o
# pythonpath do pytest.ini não tiver efeito (ex.: rodando pytest de outro cwd).
_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'scripts')
)
for _sub in ('', 'tempo_reverb', 'sti', 'claridade', 'nps'):
    _p = os.path.join(_SCRIPTS_DIR, _sub) if _sub else _SCRIPTS_DIR
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --- Fixtures básicas ---

@pytest.fixture(scope='session')
def fs_test():
    """Frequência de amostragem padrão para testes (sonômetro Classe 1)."""
    return 51200


@pytest.fixture(scope='session')
def rng():
    """Gerador determinístico para testes que precisam de ruído."""
    return np.random.default_rng(seed=42)


# --- Geradores de sinal sintético ---

def _gerar_seno(f, fs, dur, amp=1.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f * t)


@pytest.fixture
def seno():
    """Factory: seno(f, fs, dur, amp=1.0)."""
    return _gerar_seno


@pytest.fixture
def impulso_decaimento_exponencial():
    """
    Factory: gera resposta ao impulso sintética com decaimento exponencial
    em uma banda específica + ruído de fundo.

    Uso:
        h = impulso_decaimento_exponencial(fs=51200, T60=0.7, f_central=1000,
                                            dur=2.0, snr_db=50)
    """

    def _factory(fs, T60, f_central, dur=2.0, snr_db=50.0, seed=0):
        rng = np.random.default_rng(seed)
        n = int(fs * dur)
        t = np.arange(n) / fs

        # Decaimento exponencial em amplitude → T60 = 6.91 / decay_rate
        decay = np.exp(-6.908 * t / T60)

        # Componente em banda: senoide modulada por ruído branco filtrado
        # (proxy simples: senoide com fase aleatória — suficiente para testar
        # T60 sem precisar simular dispersão modal).
        carrier = rng.standard_normal(n)
        # Filtrar via FFT band-pass simples
        Y = np.fft.rfft(carrier)
        freqs = np.fft.rfftfreq(n, 1 / fs)
        band_mask = (freqs >= f_central / np.sqrt(2)) & (freqs <= f_central * np.sqrt(2))
        Y[~band_mask] = 0
        carrier = np.fft.irfft(Y, n=n)
        if np.max(np.abs(carrier)) > 0:
            carrier = carrier / np.max(np.abs(carrier))

        h = carrier * decay

        # Ruído branco aditivo (piso) com nível de snr_db abaixo do pico
        peak = np.max(np.abs(h))
        noise_amp = peak * 10 ** (-snr_db / 20)
        noise = noise_amp * rng.standard_normal(n)

        return h + noise

    return _factory
