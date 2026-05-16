# Acústica de Salas — Análise de Respostas ao Impulso

Scripts Python para análise acústica de salas de aula a partir de respostas ao impulso. Calcula T60, C50, D50, SNR, U50, STI e IF nas bandas de oitava de 125 Hz a 8000 Hz, com geração automatizada de curvas de Schroeder por banda e sala.

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
│   ├── IF.py                          # Índice de Favorabilidade
│   ├── sabine.py                      # Coeficiente de absorção via equação de Sabine
│   ├── curvas_schroeder_por_banda.py  # Curvas de Schroeder por banda de oitava
│   ├── inspecionar_ht.py              # Inspeção da resposta ao impulso em banda larga
│   ├── nps/
│   │   ├── NPS.py                     # NPS, SNR e U50 (legado, plots + console)
│   │   ├── NPS_resumo.py              # SNR + U50 por arquivo + médias → TXT
│   │   └── resultados/                # Saída do NPS_resumo
│   ├── tempo_reverb/
│   │   ├── T_reverb.py                # T20/T30/T60 com plots por banda
│   │   ├── T_reverb_resumo.py         # Variante sem plots → resultados/T60_resumo.txt
│   │   └── resultados/                # Saída do T_reverb_resumo
│   ├── sti/
│   │   ├── STI.py                     # STI do arquivo de referência (AT9_impulse_03)
│   │   ├── STI_todas_salas.py         # Itera AT5/AT9/AT10 → resultados/STI_todas_salas.txt
│   │   └── resultados/                # Saída do STI_todas_salas
│   └── claridade/
│       ├── claridade.py               # Claridade C50 e Definição D50 → TXT
│       └── resultados/                # Saída de claridade.py
├── audio/
│   └── AT{5,9,10}_*.wav               # Arquivos de áudio (ver convenção abaixo)
└── docs/
    └── *.txt                          # Documentação detalhada de cada script
```

---

## Scripts

| Script | Métricas calculadas | Saídas |
|--------|--------------------|--------------------|
| `parametros.py` | — | Funções compartilhadas, carrega `AT9_impulse_03.wav` como referência |
| `tempo_reverb/T_reverb.py` | T20, T30, T60 | PNGs `T60_{freq}Hz_{sala}.png` com curva de Schroeder e faixa dinâmica |
| `tempo_reverb/T_reverb_resumo.py` | T20, T30, T60 | TXT tabular `tempo_reverb/resultados/T60_resumo.txt` (sem plots) |
| `claridade/claridade.py` | C50, D50 | TXT `claridade/resultados/claridade_resumo.txt` por arquivo + médias por sala |
| `nps/NPS.py` | Leq, SNR, U50 (AT9 parcial) | Gráficos de NPS no tempo; console: SNR e U50 por banda |
| `nps/NPS_resumo.py` | SNR, U50 (todas as salas, s/v + c/v) | TXT `nps/resultados/NPS_resumo.txt` por arquivo + médias |
| `sti/STI.py` | MTI, STI | Console: MTI por banda, STI final e classificação qualitativa |
| `sti/STI_todas_salas.py` | MTI, STI | TXT `sti/resultados/STI_todas_salas.txt` por arquivo + médias por sala |
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
# Exemplo: calcular T60 para todas as salas (com plots)
python scripts/tempo_reverb/T_reverb.py

# Exemplo: gerar resumo tabular de T20/T30/T60 (sem plots)
python scripts/tempo_reverb/T_reverb_resumo.py
# Saída: scripts/tempo_reverb/resultados/T60_resumo.txt

# Exemplo: calcular C50 e D50
python scripts/claridade/claridade.py
# Saída: scripts/claridade/resultados/claridade_resumo.txt

# Exemplo: calcular STI do arquivo de referência
python scripts/sti/STI.py

# Exemplo: calcular STI para todos os arquivos AT5/AT9/AT10
python scripts/sti/STI_todas_salas.py
# Saída: scripts/sti/resultados/STI_todas_salas.txt

# Exemplo: SNR e U50 (sem/com ventilador) para as três salas
python scripts/nps/NPS_resumo.py
# Saída: scripts/nps/resultados/NPS_resumo.txt
```


