# Acústica de Salas — Análise de Respostas ao Impulso

Scripts Python para análise acústica de salas de aula a partir de respostas ao impulso por explosão. Calcula T60, C50, D50, SNR, U50, STI e IF nas bandas de oitava de 125 Hz a 8000 Hz, com geração automatizada de curvas de Schroeder por banda e sala.

Desenvolvido como parte da monografia *Avaliação Acústica de Salas de Aulas na Universidade Federal de São Carlos*.

---

## Ambientes estudados

| Código | Descrição | Impulsos |
|--------|-----------|----------|
| AT5    | Sala com tratamento de celulose projetada | 8 |
| AT9    | Sala sem tratamento (duas configurações) | 16 |
| AT10   | Sala com forro de fibra de vidro | 8 |

---

## Estrutura do repositório

```
.
├── scripts/
│   ├── parametros.py                  # Carregamento de áudio e funções utilitárias
│   ├── T_reverb.py                    # Tempo de reverberação T60 por banda
│   ├── claridade.py                   # Claridade C50 e Definição D50
│   ├── NPS.py                         # Nível de Pressão Sonora e SNR
│   ├── STI.py                         # Índice de Transmissão da Fala (IEC 60268-16)
│   ├── IF.py                          # Índice de Favorabilidade
│   ├── sabine.py                      # Coeficiente de absorção via equação de Sabine
│   ├── curvas_schroeder_por_banda.py  # Curvas de Schroeder por banda de oitava
│   └── inspecionar_ht.py              # Inspeção da resposta ao impulso em banda larga
├── audio/
│   └── AT{5,9,10}_*.wav               # Arquivos de áudio (ver convenção abaixo)
└── docs/
    └── *.txt                          # Documentação detalhada de cada script
```

---

## Scripts

| Script | Métricas calculadas | Saídas |
|--------|--------------------|--------------------|
| `parametros.py` | — | Funções compartilhadas, carrega `AT10_impulse_03.wav` como referência |
| `T_reverb.py` | T20, T30, T60 | PNGs `T60_{freq}Hz_{sala}.png` com curva de Schroeder e faixa dinâmica |
| `claridade.py` | C50, D50 | Console: valores por banda e por impulso; médias por ambiente |
| `NPS.py` | Leq, SNR, U50 | Gráficos de NPS no tempo; console: SNR e U50 por banda |
| `STI.py` | MTI, STI | Console: MTI por banda, STI final e classificação qualitativa |
| `IF.py` | IF | Console: índice de favorabilidade por banda (a partir de U50 pré-computado) |
| `sabine.py` | α (coeficiente de absorção) | Console: α do teto do AT10 por banda |
| `curvas_schroeder_por_banda.py` | — | PNGs `h(t)_schroeder_{freq}Hz_{sala}.png` |
| `inspecionar_ht.py` | — | Gráficos de forma de onda e energia em banda larga |

---

## Métricas acústicas

| Métrica | Descrição | Norma de referência |
|---------|-----------|---------------------|
| **T60** | Tempo de queda de 60 dB da energia acústica | ISO 3382-2 |
| **C50** | Claridade: razão log entre energia nos 50 ms iniciais e o restante (dB) | ISO 3382-1 |
| **D50** | Definição: fração da energia nos 50 ms iniciais sobre energia total (%) | ISO 3382-1 |
| **SNR** | Relação sinal-ruído por banda de oitava (dB) | — |
| **U50** | Métrica de utilidade combinando C50 e SNR (dB) | — |
| **STI** | Índice de Transmissão da Fala (0–1) | IEC 60268-16 |
| **IF** | Índice de Favorabilidade (0–100 %) via IF = −0,838·U50² + 1,027·U50 + 99,42 | — |

Todas as métricas são calculadas nas 7 bandas de oitava: **125, 250, 500, 1000, 2000, 4000 e 8000 Hz**.

---

## Convenção de nomes dos arquivos de áudio

```
AT{sala}_{tipo}_{número}.wav
```

| Campo | Valores possíveis |
|-------|-------------------|
| `sala` | `5`, `9`, `10` |
| `tipo` | `impulse`, `noise_pink`, `noise_background`, `noise_ventilator` |
| `número` | sequencial (`01`, `02`, …) ou sufixo descritivo (`mic1`, `mic2`, `atras1`) |

Exemplos:
```
AT10_impulse_03.wav
AT5_noise_pink_mic1.wav
AT9_noise_ventilator_1.wav
```

---

## Dependências

```
numpy
scipy
matplotlib
```

Instalação:

```bash
pip install numpy scipy matplotlib
```

---

## Como executar

Cada script é independente e pode ser executado diretamente. O script `parametros.py` é importado como módulo pelos demais e **não deve ser executado diretamente**.

```bash
# Exemplo: calcular T60 para todas as salas
python scripts/T_reverb.py

# Exemplo: calcular C50 e D50
python scripts/claridade.py

# Exemplo: calcular STI
python scripts/STI.py
```


