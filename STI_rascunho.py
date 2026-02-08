
# %%
# Primeiro código STI

import numpy as np
from scipy.signal import butter, sosfilt, convolve
import scipy.io.wavfile as wav

# ----------------------
# Configurações
# ----------------------
fs = 44100           # taxa de amostragem (Hz)
dur = 5.0            # duração de cada sinal (s)
m_x = 1.0            # profundidade de modulação

import parametros
h_audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t

# Frequências centrais das 7 bandas de 1 oitava (125 Hz até 8 kHz)
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]

# Frequências de modulação (14 bandas de 1 oitava de 0,63 a 12,5 Hz)
f_mod = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

# ----------------------
# Funções auxiliares
# ----------------------

def rms(signal):
    return np.sqrt(np.mean(signal**2))


# Seleção da parte correspondente ao impulso
t = np.arange(0, (len(h_audio)-1)/fs + 1/fs, 1/fs)
max_pos_linear = np.argmax(h_audio)
max_pos = np.unravel_index(max_pos_linear, h_audio.shape)
soma_indice = 28230
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
audio_cortado = h_audio[indice_inicio:indice_fim]

h_t = audio_cortado

# ----------------------
# Gerar 7 sinais anecoicos filtrados
# ----------------------
t_sin = np.linspace(0, dur, int(fs*dur), endpoint=False)
sinal = np.random.randn(len(t_sin))

sinais_anecoicos = []
for fc in f_central_lista:
    sos = parametros.filtro_banda_2(fc, fs)
    xk = sosfilt(sos, sinal)
    sinais_anecoicos.append(xk)

print(sinais_anecoicos)
print(enumerate(sinais_anecoicos))
# ----------------------
# Calcular STI com verificação para evitar NaN
# ----------------------
TI = np.zeros((len(f_central_lista), len(f_mod)))
for k, xk in enumerate(sinais_anecoicos):
    I_kx = rms(xk)**2
    for m, fm in enumerate(f_mod):
        mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sin))
        xkm = xk * mod
        ykm = convolve(xkm, h_t, mode='full')

        I_kmx = I_kx * (1 + m_x * np.cos(2*np.pi*fm*t_sin))
        I_kmy = rms(ykm)**2 * (1 + m_x * np.cos(2*np.pi*fm*np.arange(len(ykm))/fs))

        m_xkm = (2 * np.sqrt((np.sum(I_kmx * np.sin(2*np.pi*fm*t_sin)))**2 +
                             (np.sum(I_kmx * np.cos(2*np.pi*fm*t_sin)))**2)) / np.sum(I_kmx)

        m_ykm = (2 * np.sqrt((np.sum(I_kmy * np.sin(2*np.pi*fm*np.arange(len(ykm))/fs)))**2 +
                             (np.sum(I_kmy * np.cos(2*np.pi*fm*np.arange(len(ykm))/fs)))**2)) / np.sum(I_kmy)

        # Evitar divisões por zero ou valores negativos que geram log10 inválido
        if m_xkm > m_ykm and m_xkm > 0 and m_ykm >= 0:
            SNR_ef = 10 * np.log10(m_xkm / (m_xkm - m_ykm))
            TI[k, m] = (SNR_ef + 15.0) / 30.0
        else:
            TI[k, m] = 0.0

# # Média para cada banda
# MTI = np.mean(TI, axis=1)

# Média para cada banda e cálculo final do STI (equação corrigida)
TI = np.clip(TI, 0.0, 1.0)
MTI = np.mean(TI, axis=1)  # vetor de 7 valores (uma média por banda de oitava)

# Pesos alpha_k e beta_k
alpha_k = [0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14]
beta_k = [0.11, 0.12, 0.11, 0.16, 0.16, 0.14]

# Equação corrigida:
# STI = sum_k alpha_k * MTI_k - sum_k beta_k * sqrt(MTI_k * MTI_{k+1})
term1 = np.sum(alpha_k * MTI)
term2 = np.sum(beta_k * np.sqrt(MTI[:-1] * MTI[1:]))
STI = term1 + term2

print(f"STI calculado (equação corrigida): {STI:.4f}")



# %%

# ------------------------------------------------------------------------------------------------------------
# Claude sem filtro passa-baixa

import numpy as np
from scipy.signal import butter, sosfilt, convolve
import scipy.io.wavfile as wav

# ----------------------
# Configurações
# ----------------------
fs = 44100           # taxa de amostragem (Hz)
dur = 5.0            # duração de cada sinal (s)
m_x = 1.0            # profundidade de modulação

import parametros
h_audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t


# Frequências de modulação (14 bandas de 1 oitava de 0,63 a 12,5 Hz)
f_mod = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

# ----------------------
# Funções auxiliares
# ----------------------

def rms(signal):
    """Calcula o valor RMS de um sinal"""
    return np.sqrt(np.mean(signal**2))

def calcular_amplitude_modulacao(intensidade, fm, fs_signal):
    """
    Calcula a amplitude de modulação usando análise de Fourier
    baseado nas Equações 7 e 8 do texto teórico
    """
    N = len(intensidade)
    t_vec = np.arange(N) / fs_signal
    
    # Componentes seno e cosseno
    cos_comp = np.sum(intensidade * np.cos(2*np.pi*fm*t_vec))
    sin_comp = np.sum(intensidade * np.sin(2*np.pi*fm*t_vec))
    dc_comp = np.sum(intensidade)
    
    # Amplitude de modulação (Equações 7 e 8)
    if dc_comp > 0:
        m = 2 * np.sqrt(cos_comp**2 + sin_comp**2) / dc_comp
    else:
        m = 0.0
    
    return m

# Seleção da parte correspondente ao impulso
t = np.arange(0, (len(h_audio)-1)/fs + 1/fs, 1/fs)
max_pos_linear = np.argmax(h_audio)
max_pos = np.unravel_index(max_pos_linear, h_audio.shape)
soma_indice = 28230
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
audio_cortado = h_audio[indice_inicio:indice_fim]

h_t = audio_cortado

# ----------------------
# Gerar 7 sinais anecoicos filtrados
# ----------------------
t_sin = np.linspace(0, dur, int(fs*dur), endpoint=False)
sinal = np.random.randn(len(t_sin))

sinais_anecoicos = []
for i, fc in enumerate(f_central_lista):
    xk= parametros.filtro_banda(sinal, fs, fc)
    sinais_anecoicos.append(xk)
    
print("Sinais anecoicos gerados:", len(sinais_anecoicos))

# ----------------------
# Calcular STI com correções teóricas
# ----------------------
TI = np.zeros((len(f_central_lista), len(f_mod)))

for k, xk in enumerate(sinais_anecoicos):
    print(f"Processando banda {k+1}/7: {f_central_lista[k]} Hz")
    
    for m, fm in enumerate(f_mod):
        # 1. Gerar sinal anecoico modulado (Equação 1)
        mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sin))
        xkm = xk * mod
        
        # 2. Convolução com resposta ao impulso (Equação 2)
        ykm = convolve(xkm, h_t, mode='full')
        
        # 3. Calcular intensidades corretamente (Equações 3 e 4)
        # Intensidade do sinal anecoico modulado
        I_kmx_mean = rms(xkm)**2
        
        # Intensidade do sinal final modulado
        I_kmy_mean = rms(ykm)**2
        
        # 4. Calcular intensidades instantâneas para análise de modulação
        # Para o sinal anecoico modulado
        xkm_squared = xkm**2
        
        # Para o sinal final (ajustar para mesma duração para análise)
        # Truncar ou extender para manter consistência temporal
        if len(ykm) > len(xkm):
            ykm_analysis = ykm[:len(xkm)]
        else:
            ykm_analysis = np.pad(ykm, (0, len(xkm) - len(ykm)), 'constant')
        
        ykm_squared = ykm_analysis**2
        
        # 5. Calcular amplitudes de modulação (Equações 7 e 8)
        m_xkm = calcular_amplitude_modulacao(xkm_squared, fm, fs)
        m_ykm = calcular_amplitude_modulacao(ykm_squared, fm, fs)
        
        # 6. Calcular razão de modulação com termos de mascaramento
        # Para simplificação, assumindo I_ak = 0 e I_rk = 0 (sem mascaramento)
        # Em implementação completa, estes valores deveriam ser determinados
        I_ak = 0.0  # Intensidade do efeito de mascaramento
        I_rk = 0.0  # Intensidade do limiar de recepção
        
        # Razão de modulação (Equação 9 simplificada)
        if m_xkm > 0:
            m_ratio = m_ykm / m_xkm
            # Aplicar correção de intensidade se necessário
            intensity_correction = I_kmy_mean / (I_kmy_mean + I_ak + I_rk)
            m_km = m_ratio * intensity_correction
        else:
            m_km = 0.0
        
        # 7. Calcular SNR efetiva (Equação 10)
        # Evitar divisões por zero ou valores negativos
        if m_xkm > m_ykm and m_xkm > 0 and m_ykm >= 0:
            denominator = m_xkm - m_ykm
            if denominator > 1e-10:  # Evitar divisão por valores muito pequenos
                SNR_ef = 10 * np.log10(m_xkm / denominator)
            else:
                SNR_ef = -15.0  # Valor mínimo
        else:
            SNR_ef = -15.0  # Valor mínimo quando não há modulação efetiva
        
        # 8. Calcular Transmission Index (Equação 11)
        TI[k, m] = (SNR_ef + 15.0) / 30.0
        
        # Garantir que TI esteja no intervalo [0, 1]
        TI[k, m] = np.clip(TI[k, m], 0.0, 1.0)

# ----------------------
# Cálculo final do STI
# ----------------------

# Média para cada banda (Equação 12)
MTI = np.mean(TI, axis=1)  # vetor de 7 valores (uma média por banda de oitava)

print(f"MTI por banda: {MTI}")

# Pesos alpha_k e beta_k da tabela do texto
alpha_k = np.array([0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14])
beta_k = np.array([0.11, 0.12, 0.11, 0.16, 0.16, 0.14])

# Equação CORRIGIDA (Equação 13):
# STI = sum_k alpha_k * MTI_k - sum_k beta_k * sqrt(MTI_k * MTI_{k+1})
term1 = np.sum(alpha_k * MTI)
term2 = np.sum(beta_k * np.sqrt(MTI[:-1] * MTI[1:]))

# CORREÇÃO: Usar subtração conforme a teoria
STI = term1 - term2

print(f"\nResultados finais:")
print(f"Termo 1 (soma ponderada): {term1:.4f}")
print(f"Termo 2 (correção de bandas adjacentes): {term2:.4f}")
print(f"STI calculado (equação corrigida): {STI:.4f}")

# Verificar se o STI está no intervalo esperado [0, 1]
if STI < 0:
    print(f"AVISO: STI negativo ({STI:.4f}). Verificar cálculos.")
    STI = max(STI, 0.0)  # Limitar a zero se necessário
elif STI > 1:
    print(f"AVISO: STI maior que 1 ({STI:.4f}). Verificar cálculos.")
    STI = min(STI, 1.0)  # Limitar a um se necessário

print(f"STI final: {STI:.4f}")

# Interpretação do resultado
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

print(f"Qualidade da inteligibilidade da fala: {interpretacao}")

# %%
# -------------------------------------------------------------------------------------------------------------------------
# STI com filtro passa-baixa (debug)

import numpy as np
from scipy.signal import butter, sosfilt, convolve, filtfilt, hilbert
import scipy.io.wavfile as wav

# ----------------------
# Configurações
# ----------------------
fs = 44100           # taxa de amostragem (Hz)
dur = 5.0            # duração de cada sinal (s)
m_x = 1.0            # profundidade de modulação

import parametros
h_audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t

# Frequências centrais das 7 bandas de 1 oitava (125 Hz até 8 kHz)
f_central_lista = [125, 250, 500, 1000, 2000, 4000, 8000]

# Frequências de modulação (14 bandas de 1 oitava de 0,63 a 12,5 Hz)
f_mod = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

# ----------------------
# Funções auxiliares corrigidas
# ----------------------

def rms(signal):
    """Calcula o valor RMS de um sinal"""
    return np.sqrt(np.mean(signal**2))

def filtro_passa_baixa(signal, fc_max, fs_signal, order=4):
    """
    Aplica filtro passa-baixa com frequência de corte inferior a 100 Hz
    """
    fc = min(fc_max, 1000.0)  # Usar 50 Hz como frequência de corte
    nyquist = fs_signal / 2
    
    if fc >= nyquist:
        return signal
    
    try:
        sos = butter(order, fc / nyquist, btype='low', output='sos')
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal
    except Exception as e:
        print(f"ERRO no filtro: {e}")
        return signal

def calcular_intensidade_instantanea(signal, window_size=None):
    """
    Calcula a intensidade instantânea real do sinal usando envelope de Hilbert
    ou janela deslizante, em vez de assumir forma cossenoidal
    """
    if window_size is None:
        # Usar transformada de Hilbert para envelope
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        intensity = envelope**2
    else:
        # Método alternativo: janela deslizante para RMS
        intensity = np.zeros_like(signal)
        half_window = window_size // 2
        
        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window + 1)
            intensity[i] = np.mean(signal[start:end]**2)
    
    return intensity

def calcular_amplitude_modulacao_fft(intensity_envelope, fm, fs_signal):
    """
    Calcula amplitude de modulação usando FFT para ser mais preciso
    """
    N = len(intensity_envelope)
    
    # Remover componente DC
    intensity_ac = intensity_envelope - np.mean(intensity_envelope)
    dc_component = np.mean(intensity_envelope)
    
    if dc_component <= 0:
        return 0.0
    
    # Calcular FFT
    fft_result = np.fft.fft(intensity_ac)
    freqs = np.fft.fftfreq(N, 1/fs_signal)
    
    # Encontrar o bin de frequência mais próximo de fm
    fm_idx = np.argmin(np.abs(freqs - fm))
    
    # Amplitude na frequência de modulação
    amplitude_complex = fft_result[fm_idx]
    amplitude = 2 * np.abs(amplitude_complex) / N
    
    # Amplitude de modulação normalizada
    m = amplitude / dc_component
    
    return m

def calcular_amplitude_modulacao_correlacao(intensity_envelope, fm, fs_signal):
    """
    Método baseado em correlação com seno e cosseno (mais robusto)
    """
    N = len(intensity_envelope)
    t_vec = np.arange(N) / fs_signal
    
    # Remover DC
    mean_intensity = np.mean(intensity_envelope)
    if mean_intensity <= 0:
        return 0.0
    
    intensity_ac = intensity_envelope - mean_intensity
    
    # Correlação com cosseno e seno
    cos_ref = np.cos(2*np.pi*fm*t_vec)
    sin_ref = np.sin(2*np.pi*fm*t_vec)
    
    # Calcular correlações normalizadas
    cos_corr = 2 * np.mean(intensity_ac * cos_ref)
    sin_corr = 2 * np.mean(intensity_ac * sin_ref)
    
    # Amplitude de modulação
    m = np.sqrt(cos_corr**2 + sin_corr**2) / mean_intensity
    
    return m

# Seleção da parte correspondente ao impulso
t = np.arange(0, (len(h_audio)-1)/fs + 1/fs, 1/fs)
max_pos_linear = np.argmax(h_audio)
max_pos = np.unravel_index(max_pos_linear, h_audio.shape)
soma_indice = 28230
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
audio_cortado = h_audio[indice_inicio:indice_fim]

h_t = audio_cortado

print(f"Resposta ao impulso: len={len(h_t)}, max={np.max(np.abs(h_t)):.6f}")


# Geração de sinais anecoicos
t_sin = np.linspace(0, dur, int(fs*dur), endpoint=False)
sinal = np.random.randn(len(t_sin))

sinais_anecoicos = []
for i, fc in enumerate(f_central_lista):
    xk= parametros.filtro_banda(sinal, fs, fc)
    sinais_anecoicos.append(xk)

# ----------------------
# TESTE DETALHADO CORRIGIDO
# ----------------------
k_test = 0  # Primeira banda (125 Hz)
m_test = 6  # Frequência de modulação (2.5 Hz)

xk = sinais_anecoicos[k_test]
fm = f_mod[m_test]

print(f"\n=== TESTE CORRIGIDO - Banda {f_central_lista[k_test]} Hz, Mod {fm} Hz ===")

# 1. Sinal anecoico modulado
mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sin))
xkm = xk * mod
print(f"1. Sinal modulado: rms={rms(xkm):.6f}")

# 2. Convolução
ykm = convolve(xkm, h_t, mode='full')
if len(ykm) > len(xkm):
    ykm_analysis = ykm[:len(xkm)]
else:
    ykm_analysis = np.pad(ykm, (0, len(xkm) - len(ykm)), 'constant')

print(f"2. Após convolução: rms={rms(ykm_analysis):.6f}")

# 3. Calcular intensidades instantâneas REAIS
print("3. Calculando intensidades instantâneas reais...")

# Para o sinal modulado original
I_kmx_real = calcular_intensidade_instantanea(xkm)
print(f"   I_kmx_real: min={np.min(I_kmx_real):.6f}, max={np.max(I_kmx_real):.6f}, mean={np.mean(I_kmx_real):.6f}")

# Para o sinal após convolução
I_kmy_real = calcular_intensidade_instantanea(ykm_analysis)
print(f"   I_kmy_real: min={np.min(I_kmy_real):.6f}, max={np.max(I_kmy_real):.6f}, mean={np.mean(I_kmy_real):.6f}")

# 4. Aplicar filtro passa-baixa nas intensidades reais
I_kmx_envelope = filtro_passa_baixa(I_kmx_real, 100.0, fs)
I_kmy_envelope = filtro_passa_baixa(I_kmy_real, 100.0, fs)

print(f"4. Envoltórias filtradas:")
print(f"   I_kmx_env: min={np.min(I_kmx_envelope):.6f}, max={np.max(I_kmx_envelope):.6f}")
print(f"   I_kmy_env: min={np.min(I_kmy_envelope):.6f}, max={np.max(I_kmy_envelope):.6f}")

# 5. Calcular amplitudes de modulação usando métodos corretos
print(f"5. Calculando amplitudes de modulação:")

# Método por correlação
m_xkm_corr = calcular_amplitude_modulacao_correlacao(I_kmx_envelope, fm, fs)
m_ykm_corr = calcular_amplitude_modulacao_correlacao(I_kmy_envelope, fm, fs)

print(f"   Correlação: m_x = {m_xkm_corr:.6f}, m_y = {m_ykm_corr:.6f}")

# Método por FFT
m_xkm_fft = calcular_amplitude_modulacao_fft(I_kmx_envelope, fm, fs)
m_ykm_fft = calcular_amplitude_modulacao_fft(I_kmy_envelope, fm, fs)

print(f"   FFT: m_x = {m_xkm_fft:.6f}, m_y = {m_ykm_fft:.6f}")

# Usar método por correlação (geralmente mais robusto)
m_xkm = m_xkm_corr
m_ykm = m_ykm_corr

# 6. Calcular SNR efetiva
print(f"6. Cálculo SNR:")
print(f"   m_x = {m_xkm:.6f}, m_y = {m_ykm:.6f}")

if m_xkm > m_ykm and m_xkm > 1e-6 and m_ykm >= 0:
    denominator = m_xkm - m_ykm
    print(f"   Denominador = {denominator:.6f}")
    
    if denominator > 1e-6:
        SNR_ef = 10 * np.log10(m_xkm / denominator)
        print(f"   SNR efetiva: {SNR_ef:.4f} dB")
        TI_val = np.clip((SNR_ef + 15.0) / 30.0, 0.0, 1.0)
        print(f"   TI = {TI_val:.6f}")
    else:
        print(f"   Denominador muito pequeno")
        TI_val = 0.0
else:
    print(f"   Condição inválida para SNR")
    TI_val = 0.0

print(f"\n=== PROCESSAMENTO COMPLETO ===")

# ----------------------
# Calcular STI completo com método corrigido
# ----------------------
TI = np.zeros((len(f_central_lista), len(f_mod)))

for k, xk in enumerate(sinais_anecoicos):
    print(f"Processando banda {k+1}/7: {f_central_lista[k]} Hz")
    
    for m, fm in enumerate(f_mod):
        # Gerar sinal modulado
        mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sin))
        xkm = xk * mod
        
        # Convolução
        ykm = convolve(xkm, h_t, mode='full')
        
        # Ajustar comprimento
        if len(ykm) > len(xkm):
            ykm_analysis = ykm[:len(xkm)]
        else:
            ykm_analysis = np.pad(ykm, (0, len(xkm) - len(ykm)), 'constant')
        
        # Calcular intensidades instantâneas reais
        I_kmx_real = calcular_intensidade_instantanea(xkm)
        I_kmy_real = calcular_intensidade_instantanea(ykm_analysis)
        
        # Aplicar filtro passa-baixa
        I_kmx_envelope = filtro_passa_baixa(I_kmx_real, 100.0, fs)
        I_kmy_envelope = filtro_passa_baixa(I_kmy_real, 100.0, fs)
        
        # Calcular amplitudes de modulação
        m_xkm = calcular_amplitude_modulacao_correlacao(I_kmx_envelope, fm, fs)
        m_ykm = calcular_amplitude_modulacao_correlacao(I_kmy_envelope, fm, fs)
        
        # Calcular SNR e TI
        if m_xkm > m_ykm and m_xkm > 1e-6 and m_ykm >= 0:
            denominator = m_xkm - m_ykm
            if denominator > 1e-6:
                SNR_ef = 10 * np.log10(m_xkm / denominator)
                TI[k, m] = np.clip((SNR_ef + 15.0) / 30.0, 0.0, 1.0)
            else:
                TI[k, m] = 0.0
        else:
            TI[k, m] = 0.0

# Cálculo final
MTI = np.mean(TI, axis=1)
print(f"\nMTI por banda: {MTI}")

# Verificar se temos valores válidos
if np.all(MTI == 0):
    print("AINDA TODOS ZEROS - vamos tentar abordagem alternativa...")
    
    # Método alternativo mais direto
    print("Testando método direto baseado na energia...")
    
    for k, xk in enumerate(sinais_anecoicos):
        for m, fm in enumerate(f_mod):
            # Método simplificado: comparar energia antes e depois
            mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sin))
            xkm = xk * mod
            
            ykm = convolve(xkm, h_t, mode='full')
            if len(ykm) > len(xkm):
                ykm = ykm[:len(xkm)]
            
            # Energia dos sinais
            E_x = np.var(xkm)  # Variância como medida de energia AC
            E_y = np.var(ykm)
            
            if E_x > 0:
                # Fator de transferência de modulação
                transfer_factor = E_y / E_x
                
                # Simular degradação da modulação
                degradation = min(transfer_factor, 1.0)
                
                if degradation > 0.1:  # Threshold mínimo
                    SNR_ef = 10 * np.log10(1.0 / (1.0 - degradation + 1e-6))
                    TI[k, m] = np.clip((SNR_ef + 15.0) / 30.0, 0.0, 1.0)

    MTI = np.mean(TI, axis=1)
    print(f"MTI método alternativo: {MTI}")

# Cálculo final do STI
alpha_k = np.array([0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14])
beta_k = np.array([0.11, 0.12, 0.11, 0.16, 0.16, 0.14])*0.1

print(alpha_k * MTI)
print(beta_k * np.sqrt(MTI[1:] * MTI[:-1]))

term1 = np.sum(alpha_k * MTI)
term2 = np.sum(beta_k * np.sqrt(MTI[1:] * MTI[:-1]))
STI = term1 - term2

print(f"\nResultados finais:")
print(f"Termo 1 (soma ponderada): {term1:.4f}")
print(f"Termo 2 (correção de bandas adjacentes): {term2:.4f}")
print(f"STI = {STI:.4f}")

# Interpretação
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

# -------------------------------------------------------------------------------------------------------------------------
# STI com filtro passa-baixa (simplificado)

import numpy as np
from scipy.signal import butter, sosfilt, convolve, filtfilt, hilbert
import scipy.io.wavfile as wav

import parametros
h_audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t
dur = 5.0            # duração de cada sinal (s)
m_x = 1.0            # profundidade de modulação

# Frequências de modulação
f_mod = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

# Seleção da parte correspondente ao impulso
t = np.arange(0, (len(h_audio)-1)/fs + 1/fs, 1/fs)
max_pos_linear = np.argmax(h_audio)
max_pos = np.unravel_index(max_pos_linear, h_audio.shape)
soma_indice = 28230
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
audio_cortado = h_audio[indice_inicio:indice_fim]

h_t = audio_cortado

# Geração de sinais anecoicos
t_sin = np.linspace(0, dur, int(fs*dur), endpoint=False)
sinal = np.random.randn(len(t_sin))

sinais_anecoicos = []
for i, fc in enumerate(f_central_lista):
    xk= parametros.filtro_banda(sinal, fs, fc)
    sinais_anecoicos.append(xk)


# Calculo do STI
TI = np.zeros((len(f_central_lista), len(f_mod)))

for k, xk in enumerate(sinais_anecoicos):
    print(f"Processando banda {k+1}/7: {f_central_lista[k]} Hz")
    
    for m, fm in enumerate(f_mod):
        # Gerar sinal modulado
        mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sin))
        xkm = xk * mod
        
        # Convolução
        ykm = convolve(xkm, h_t, mode='full')
        
        # Ajustar comprimento
        if len(ykm) > len(xkm):
            ykm_analysis = ykm[:len(xkm)]
        else:
            ykm_analysis = np.pad(ykm, (0, len(xkm) - len(ykm)), 'constant')
        
        # Calcular intensidades instantâneas reais
        I_kmx_real = parametros.calcular_intensidade_instantanea(xkm)
        I_kmy_real = parametros.calcular_intensidade_instantanea(ykm_analysis)
        
        # Aplicar filtro passa-baixa
        I_kmx_envelope = parametros.filtro_passa_baixa(I_kmx_real, 100.0, fs)
        I_kmy_envelope = parametros.filtro_passa_baixa(I_kmy_real, 100.0, fs)
        
        # Calcular amplitudes de modulação
        m_xkm = parametros.calcular_amplitude_modulacao_correlacao(I_kmx_envelope, fm, fs)
        m_ykm = parametros.calcular_amplitude_modulacao_correlacao(I_kmy_envelope, fm, fs)
        
        # Calcular SNR e TI
        if m_xkm > m_ykm and m_xkm > 1e-6 and m_ykm >= 0:
            denominator = m_xkm - m_ykm
            if denominator > 1e-6:
                SNR_ef = 10 * np.log10(m_xkm / denominator)
                TI[k, m] = np.clip((SNR_ef + 15.0) / 30.0, 0.0, 1.0)
            else:
                TI[k, m] = 0.0
        else:
            TI[k, m] = 0.0

MTI = np.mean(TI, axis=1)
print(f"\nMTI por banda: {MTI}")

# Cálculo final do STI
alpha_k = np.array([0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14])
beta_k = np.array([0.11, 0.12, 0.11, 0.16, 0.16, 0.14])

term1 = np.sum(alpha_k * MTI)
term2 = np.sum(beta_k * np.sqrt(MTI[1:] * MTI[:-1]))
STI = term1 - term2

print(f"\nResultados finais:")
print(f"Termo 1 (peso de banda central): {term1:.4f}")
print(f"Termo 2 (peso de bandas adjacentes): {term2:.4f}")
print(f"STI = {STI:.4f}")

# Interpretação
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
# --------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.signal import butter, sosfilt, convolve, filtfilt, hilbert
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import parametros
h_audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t
dur = 5.0            # duração de cada sinal (s)
m_x = 1.0            # profundidade de modulação

# Frequências de modulação
f_mod = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

# Seleção da parte correspondente ao impulso
t = np.arange(0, (len(h_audio)-1)/fs + 1/fs, 1/fs)
max_pos_linear = np.argmax(h_audio)
max_pos = np.unravel_index(max_pos_linear, h_audio.shape)
soma_indice = 28230 -1
indice_inicio = max_pos[0] - 600
indice_fim = max_pos[0] + soma_indice
t_inicio = t[indice_inicio]
t_fim = t[indice_fim]
audio_cortado = h_audio[indice_inicio:indice_fim]

h_t = audio_cortado
t_cortado = np.arange(t_inicio, t_fim, 1/fs)

# Geração de sinais anecoicos
t_sin = np.linspace(0, dur, int(fs*dur), endpoint=False)
sinal = np.random.randn(len(t_sin))

sinais_anecoicos = []
for i, fc in enumerate(f_central_lista):
    xk= parametros.filtro_banda(sinal, fs, fc)
    sinais_anecoicos.append(xk)


plt.figure()
plt.plot(t_cortado, h_t, color="dodgerblue")
plt.xlabel("Tempo [s]")
plt.ylabel("Pressão sonora")
# plt.title(f"Curva de decaimento")
plt.grid(True)
plt.ylim(-1, 1)
# plt.savefig(f"h_t.png")
for i in range(0,6,1):
    plt.plot(t_sin, sinais_anecoicos[i], color="dodgerblue")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Pressão sonora")
    # plt.title(f"Curva de decaimento")
    plt.grid(True)
    plt.ylim(-2, 2)
    plt.xlim(0, 5)
    plt.tight_layout()
    # plt.savefig(f"curva_decaimento.png")
    plt.title(f"{f_central_lista[i]} Hz")
    # plt.savefig(f"sinais_anacoicos_{int(f_central_lista[i])}Hz.png")
    plt.show()

# Calculo do STI
TI = np.zeros((len(f_central_lista), len(f_mod)))

for k, xk in enumerate(sinais_anecoicos):
    print(f"Processando banda {k+1}/7: {f_central_lista[k]} Hz")
    k=3
    xk=sinais_anecoicos[3]
    for m, fm in enumerate(f_mod):
        # Gerar sinal modulado
        mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sin))
        xkm = xk * mod
        t_x=np.arange(0, parametros.t[len(xkm)], 1/parametros.fs)

        plt.plot(t_x, xkm, color="dodgerblue")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Pressão sonora")
        plt.grid(True)
        # plt.ylim(-2, 2)
        plt.xlim(0, 6)
        plt.tight_layout()
        plt.title(f"{fm} Hz")
        # plt.savefig(f"sinais_convolução_{fm}Hz.png")
        plt.show()
        
        # Convolução
        ykm = convolve(xkm, h_t, mode='full')

        t_y=np.arange(0, parametros.t[len(ykm)], 1/parametros.fs)

        # plt.plot(t_y, ykm, color="dodgerblue")
        # plt.xlabel("Tempo [s]")
        # plt.ylabel("Pressão sonora")
        # plt.grid(True)
        # # plt.ylim(-2, 2)
        # plt.xlim(0, 6)
        # plt.tight_layout()
        # plt.title(f"{fm} Hz")
        # # plt.savefig(f"sinais_convolução_{fm}Hz.png")
        # plt.show()

        # Ajustar comprimento
        if len(ykm) > len(xkm):
            ykm_analysis = ykm[:len(xkm)]
        else:
            ykm_analysis = np.pad(ykm, (0, len(xkm) - len(ykm)), 'constant')
        
        # Calcular intensidades instantâneas reais
        I_kmx_real = parametros.calcular_intensidade_instantanea(xkm)
        I_kmy_real = parametros.calcular_intensidade_instantanea(ykm_analysis)
        
        # Aplicar filtro passa-baixa
        I_kmx_envelope = parametros.filtro_passa_baixa(I_kmx_real, 100.0, fs)
        I_kmy_envelope = parametros.filtro_passa_baixa(I_kmy_real, 100.0, fs)
        
        # Calcular amplitudes de modulação
        m_xkm = parametros.calcular_amplitude_modulacao_correlacao(I_kmx_envelope, fm, fs)
        m_ykm = parametros.calcular_amplitude_modulacao_correlacao(I_kmy_envelope, fm, fs)
        
        # Calcular SNR e TI
        if m_xkm > m_ykm and m_xkm > 1e-6 and m_ykm >= 0:
            denominator = m_xkm - m_ykm
            if denominator > 1e-6:
                SNR_ef = 10 * np.log10(m_xkm / denominator)
                TI[k, m] = np.clip((SNR_ef + 15.0) / 30.0, 0.0, 1.0)
            else:
                TI[k, m] = 0.0
        else:
            TI[k, m] = 0.0

MTI = np.mean(TI, axis=1)
print(f"\nMTI por banda: {MTI}")

# Cálculo final do STI
alpha_k = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
beta_k = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])

term1 = np.sum(alpha_k * MTI)
term2 = np.sum(beta_k * np.sqrt(MTI[1:] * MTI[1:]))
STI = term1 - term2

print(f"\nResultados finais:")
print(f"Termo 1 (peso de banda central): {term1:.4f}")
print(f"Termo 2 (peso de bandas adjacentes): {term2:.4f}")
print(f"STI = {STI:.4f}")

# Interpretação
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
