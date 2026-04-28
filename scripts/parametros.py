"""parametros.py — Funções utilitárias compartilhadas para análise acústica."""

import os
import numpy as np
from scipy.signal import butter, sosfilt, convolve, filtfilt, hilbert
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Frequências centrais das bandas de oitava (ISO 266)
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]
fs_target = 51200  # Frequência de amostragem típica do sonômetro Classe 1

_AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio')
_DEFAULT_AUDIO = os.path.join(_AUDIO_DIR, 'AT9_impulse_03.wav')


# --- Carregamento do áudio padrão ---
def load_default_audio():
    """
    Carrega o .wav de referência, normaliza e gera a janela centrada no pico.
    Retorna dict com fs, audio, t, audio_cortado, t_cortado.
    """
    fs, audio_raw = wav.read(_DEFAULT_AUDIO)
    audio = audio_raw / np.max(np.abs(audio_raw))
    t = np.arange(0, (len(audio) - 1) / fs + 1 / fs, 1 / fs)

    max_pos = int(np.argmax(audio))
    soma_indice = 28230 - 1
    indice_inicio = max_pos - 600
    indice_fim = max_pos + soma_indice
    audio_cortado = audio[indice_inicio:indice_fim]
    t_cortado = np.arange(t[indice_inicio], t[indice_fim], 1 / fs)

    return {
        'fs': fs,
        'audio': audio,
        't': t,
        'audio_cortado': audio_cortado,
        't_cortado': t_cortado,
    }


# --- Filtro de banda de oitava ---
def filtro_banda(data, fs, f_centro):
    """
    Filtro Butterworth passa-banda de ordem 4 em SOS, aplicado em modo zero-phase
    via sosfiltfilt. Limites da banda seguem razão sqrt(2) (banda de oitava).

    Args:
        data: array 1D do sinal a filtrar.
        fs: frequência de amostragem em Hz.
        f_centro: frequência central da banda em Hz.

    Returns:
        array 1D filtrado, mesmo comprimento de data.

    Raises:
        ValueError: se a frequência superior da banda excede Nyquist.
    """
    f1 = f_centro / np.sqrt(2)
    f2 = f_centro * np.sqrt(2)

    if f2 >= fs / 2:
        raise ValueError(
            f"Banda {f_centro} Hz: limite superior f2={f2:.1f} Hz "
            f">= Nyquist={fs/2:.1f} Hz. Aumente fs ou remova esta banda."
        )

    sos = signal.butter(4, [f1, f2], btype='band', fs=fs, output='sos')

    # padlen >= 3 ciclos da menor frequência da banda
    padlen_desejado = max(3 * int(fs / f1), 3 * 4 * 2)
    padlen = min(padlen_desejado, len(data) - 1)

    return signal.sosfiltfilt(sos, data, padlen=padlen)


def rms(s):
    return np.sqrt(np.mean(s ** 2))


def filtro_passa_baixa(signal_in, fc, fs_signal, order=4):
    """
    Filtro passa-baixa Butterworth de ordem `order`, aplicado em modo zero-phase.

    Args:
        signal_in: array 1D do sinal de entrada.
        fc: frequência de corte em Hz (sem cap interno).
        fs_signal: frequência de amostragem em Hz.
        order: ordem do filtro (default 4).

    Returns:
        array 1D filtrado, mesmo comprimento de signal_in.
        Se fc >= Nyquist, retorna signal_in inalterado.
    """
    nyquist = fs_signal / 2
    if fc >= nyquist:
        return signal_in

    sos = butter(order, fc / nyquist, btype='low', output='sos')
    padlen = min(3 * order * 2, len(signal_in) - 1)
    return signal.sosfiltfilt(sos, signal_in, padlen=padlen)


def lundeby_truncation_point(energy, fs, max_iter=5):
    """
    Encontra o ponto de truncamento da integral de Schroeder onde a curva de
    decaimento intersecta o piso de ruído, conforme método iterativo de Lundeby
    (Lundeby et al., 1995; ISO 3382-2 Anexo).

    Args:
        energy: array 1D da energia instantânea (h(t)^2).
        fs: frequência de amostragem em Hz.
        max_iter: número máximo de iterações de refinamento (default 5).

    Returns:
        Índice (int) do ponto de truncamento. Se o algoritmo não convergir ou
        os dados forem insuficientes, retorna len(energy) (i.e., não trunca).
    """
    energy = np.asarray(energy, dtype=float)
    N = len(energy)
    if N < int(0.5 * fs):  # menos de 0,5 s — não vale a pena
        return N

    # 1. Estimativa inicial do piso de ruído pelos últimos 10% do sinal.
    tail_start = int(0.9 * N)
    noise_floor = np.mean(energy[tail_start:])
    if noise_floor <= 0:
        return N

    # 2. Suavizar energia em janelas de ~30 ms (típico em literatura).
    win_ms = 30
    win = max(int(fs * win_ms / 1000), 1)
    kernel = np.ones(win) / win
    smoothed = np.convolve(energy, kernel, mode='same')
    smoothed = np.maximum(smoothed, 1e-30)
    smoothed_db = 10 * np.log10(smoothed)
    noise_db = 10 * np.log10(max(noise_floor, 1e-30))

    # 3. Ajuste linear preliminar entre o pico e (piso + 10 dB).
    upper = np.max(smoothed_db[: max(int(0.05 * N), 1)])  # topo nas primeiras 5%
    target = noise_db + 10
    mask = (smoothed_db <= upper) & (smoothed_db >= target)
    if mask.sum() < 2:
        return N

    t_arr = np.arange(N) / fs
    slope, intercept = np.polyfit(t_arr[mask], smoothed_db[mask], 1)
    if slope >= 0:  # decaimento inválido — sinal não decai
        return N

    # 4. Iteração: refinar estimativa do piso, intervalo de regressão e cruzamento.
    crosspoint_idx = N
    for _ in range(max_iter):
        # cruzamento entre reta ajustada e piso de ruído
        crosspoint_t = (noise_db - intercept) / slope
        idx_new = int(np.clip(crosspoint_t * fs, 1, N - 1))

        # novo piso: medido a partir de ~10% do sinal além do cruzamento
        new_tail_start = min(idx_new + int(0.1 * fs), N - 1)
        if new_tail_start >= N - 2:
            crosspoint_idx = idx_new
            break
        new_noise_floor = np.mean(energy[new_tail_start:])
        if new_noise_floor <= 0:
            crosspoint_idx = idx_new
            break
        new_noise_db = 10 * np.log10(new_noise_floor)

        # novo intervalo de regressão: entre topo e (novo_piso + 5 dB)
        new_target = new_noise_db + 5
        mask = (smoothed_db <= upper) & (smoothed_db >= new_target)
        if mask.sum() < 2:
            crosspoint_idx = idx_new
            break

        new_slope, new_intercept = np.polyfit(t_arr[mask], smoothed_db[mask], 1)
        if new_slope >= 0:
            crosspoint_idx = idx_new
            break

        # critério de convergência: variação < 10 ms
        converged = abs(idx_new - crosspoint_idx) < int(0.01 * fs)
        slope, intercept, noise_db = new_slope, new_intercept, new_noise_db
        crosspoint_idx = idx_new
        if converged:
            break

    return crosspoint_idx


def ajustar_t60(t_sch, schroeder_db):
    """
    Ajuste linear da curva de Schroeder no intervalo [-5 dB, -25 dB] e cálculo
    de T20, T30 e T60 a partir do slope e intercept.

    Args:
        t_sch: array 1D dos tempos.
        schroeder_db: array 1D da curva de Schroeder em dB (normalizada).

    Returns:
        Tupla (coeffs, t_fit, T20, T30, T60) ou None se houver dados
        insuficientes na faixa de ajuste.
    """
    mask = (schroeder_db <= -5) & (schroeder_db >= -25)
    t_fit = t_sch[mask]
    y_fit = schroeder_db[mask]
    if len(t_fit) <= 1:
        return None
    coeffs = np.polyfit(t_fit, y_fit, 1)
    if coeffs[0] >= 0:
        return None
    T20 = (-20 - coeffs[1]) / coeffs[0]
    T30 = (-30 - coeffs[1]) / coeffs[0]
    T60 = (-60 - coeffs[1]) / coeffs[0]
    return coeffs, t_fit, T20, T30, T60


def plot_filter_response(f_centro, fs, ax=None):
    """
    Plota a resposta de magnitude efetiva do filtro de banda em torno de f_centro
    (considerando o efeito do filtfilt — magnitude ao quadrado). Útil para
    verificar conformidade com a máscara IEC 61260 Classe 1.

    Args:
        f_centro: frequência central da banda em Hz.
        fs: frequência de amostragem em Hz.
        ax: eixo matplotlib opcional; se None, cria figura nova.

    Returns:
        Tupla (w, h_efetiva_db) com as frequências e magnitudes em dB.
    """
    f1 = f_centro / np.sqrt(2)
    f2 = f_centro * np.sqrt(2)
    sos = signal.butter(4, [f1, f2], btype='band', fs=fs, output='sos')
    w, h = signal.sosfreqz(sos, worN=4096, fs=fs)
    h_efetiva_db = 20 * np.log10(np.abs(h) + 1e-12) * 2  # *2 = efeito do filtfilt

    if ax is None:
        fig, ax = plt.subplots()
    ax.semilogx(w, h_efetiva_db)
    ax.axhline(-3, color='gray', ls=':', label='-3 dB')
    ax.axvline(f1, color='gray', ls=':')
    ax.axvline(f2, color='gray', ls=':')
    ax.set_title(f'Resposta de magnitude — {f_centro} Hz (efetiva pós-filtfilt)')
    ax.set_xlabel('Frequência [Hz]')
    ax.set_ylabel('|H(f)| [dB]')
    ax.set_ylim(-120, 5)
    ax.grid(True, which='both')
    ax.legend()
    return w, h_efetiva_db


def calcular_intensidade_instantanea(s, window_size=None):
    """
    Calcula a intensidade instantânea do sinal usando envelope de Hilbert
    (default) ou janela deslizante para RMS.
    """
    if window_size is None:
        analytic_signal = hilbert(s)
        envelope = np.abs(analytic_signal)
        intensity = envelope ** 2
    else:
        intensity = np.zeros_like(s, dtype=float)
        half_window = window_size // 2
        for i in range(len(s)):
            start = max(0, i - half_window)
            end = min(len(s), i + half_window + 1)
            intensity[i] = np.mean(s[start:end] ** 2)
    return intensity


def calcular_amplitude_modulacao_fft(intensity_envelope, fm, fs_signal):
    """Amplitude de modulação via FFT (mais preciso para sinais longos)."""
    N = len(intensity_envelope)
    intensity_ac = intensity_envelope - np.mean(intensity_envelope)
    dc_component = np.mean(intensity_envelope)
    if dc_component <= 0:
        return 0.0
    fft_result = np.fft.fft(intensity_ac)
    freqs = np.fft.fftfreq(N, 1 / fs_signal)
    fm_idx = np.argmin(np.abs(freqs - fm))
    amplitude_complex = fft_result[fm_idx]
    amplitude = 2 * np.abs(amplitude_complex) / N
    return amplitude / dc_component


def calcular_amplitude_modulacao_correlacao(intensity_envelope, fm, fs_signal):
    """Amplitude de modulação via correlação com seno e cosseno (mais robusto)."""
    N = len(intensity_envelope)
    t_vec = np.arange(N) / fs_signal
    mean_intensity = np.mean(intensity_envelope)
    if mean_intensity <= 0:
        return 0.0
    intensity_ac = intensity_envelope - mean_intensity
    cos_ref = np.cos(2 * np.pi * fm * t_vec)
    sin_ref = np.sin(2 * np.pi * fm * t_vec)
    cos_corr = 2 * np.mean(intensity_ac * cos_ref)
    sin_corr = 2 * np.mean(intensity_ac * sin_ref)
    return np.sqrt(cos_corr ** 2 + sin_corr ** 2) / mean_intensity


def L_equivalente(audio, p_ref):
    return 10 * np.log10(np.mean((audio / p_ref) ** 2))


def L_max(audio, p_ref):
    p_max = np.max(np.abs(audio))
    return 20 * np.log10(p_max / p_ref)


def audio_pascal(audio, fs=None):
    """
    Conversão do sinal digital normalizado para pressão sonora (Pa).
    Sensibilidade do mic: 50 mV/Pa; full-scale: 1 V (±1 V = 2 Vpp).

    Args:
        audio: array 1D do sinal.
        fs: frequência de amostragem (necessária só para o corte em 121 s).
            Se None, usa fs_target (51 200 Hz).

    Returns:
        array 1D em Pascal.
    """
    if fs is None:
        fs = fs_target

    mic_sensitivity = 50e-3      # 50 mV/Pa
    full_scale_voltage = 1.0     # ±1 V
    max_duration = 121

    max_samples = int(fs * max_duration)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    if audio.dtype == np.int16:
        max_val = 2 ** 15
    elif audio.dtype == np.int32:
        max_val = 2 ** 31
    else:
        max_val = np.max(np.abs(audio))

    audio_norm = audio / max_val
    return (audio_norm * full_scale_voltage) / mic_sensitivity


if __name__ == "__main__":
    d = load_default_audio()
    print(f"fs={d['fs']} Hz, len(audio)={len(d['audio'])}")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for f, ax in zip([125, 1000, 8000], axes):
        plot_filter_response(f, d['fs'], ax=ax)
    plt.tight_layout()
    plt.show()
