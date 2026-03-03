
import numpy as np
from scipy.signal import butter, sosfilt, convolve, filtfilt, hilbert
import scipy.io.wavfile as wav

import parametros
h_audio = parametros.audio
fs = parametros.fs
f_central_lista = parametros.f_central_lista
t = parametros.t
dur = 5.0          # duração de cada sinal (s)
m_x = 1.0            # profundidade de modulação

# Frequências de modulação
f_mod = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5]

if __name__ == "__main__":
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
    t_sig = np.linspace(0, dur, int(fs*dur), endpoint=False)
    noise = np.random.randn(len(t_sig))

    sinais_anecoicos = []
    for i, fc in enumerate(f_central_lista):
        xk= parametros.filtro_banda(noise, fs, fc)
        sinais_anecoicos.append(xk)


    # Calculo do STI
    TI = np.zeros((len(f_central_lista), len(f_mod)))

    for k, xk in enumerate(sinais_anecoicos):
        print(f"Processando banda {k+1}/7: {f_central_lista[k]} Hz")

        for m, fm in enumerate(f_mod):
            # Gerar sinal modulado
            mod = np.sqrt(1 + m_x * np.cos(2*np.pi*fm*t_sig))
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
