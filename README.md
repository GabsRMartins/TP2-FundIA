# TP2 — Busca Local e Algoritmo Genético

Fundamentos de Inteligência Artificial (UFMG). O trabalho resolve o **sequenciamento de manutenção de
ativos**: encontrar a ordem de visita que minimiza o atraso ponderado total
h(π) = Σ w_j · max(0, C_j − d_j). Três algoritmos são comparados sob o mesmo orçamento de avaliações de h:

- **GD**: greedy descent com random restart, nas variantes best improvement, first improvement e
  sucessor aleatório;
- **SA**: simulated annealing com schedule geométrico calibrado por instância;
- **GA**: genetic algorithm com torneio, Order Crossover (OX), mutação e elitismo.

## Requisitos

Python 3.10+ (desenvolvido com 3.12) e as bibliotecas de `requirements.txt`:

```bash
pip install -r requirements.txt
```

Todos os comandos abaixo são executados **dentro desta pasta** (`TP2/`).

## Estrutura

```
manutencao_{pequena,media,grande}.txt  instâncias do enunciado
manutencao_exemplo.txt                 instância da Figura 1 (n = 4; coordenadas deduzidas do cronograma)
entities/   Ativo, Instancia (leitura do arquivo, Eq. 1, h de referência) e Movimento
services/   interface.py  ProblemaPermutacao, a interface comum da Seção 2.1
            problema.py   ProblemaManutencao: operadores e avaliação incremental
            executor.py   protocolo experimental, sementes, execução paralela, gravação em disco
            analise.py    tabelas e testes estatísticos a partir de resultados/
utils/      vizinhancas.py (swap, inserção, 2-opt) · cruzamentos.py (OX, PMX)
            busca_local.py (GD, SA) · genetico.py (GA) · metricas.py (contagem de avaliações)
            estatistica.py (Mann–Whitney, Holm, A12) · referencia.py (guloso e ótimo exato)
tests/      testes automáticos (unittest)
main.py     linha de comando dos experimentos
gerar_figuras.py   figuras do relatório a partir de resultados/
resultados/ CSV (uma linha por execução) e JSON (curvas e séries) gravados pelos experimentos
Figuras/    PNG gerados por gerar_figuras.py
```

Os algoritmos (`utils/busca_local.py`, `utils/genetico.py`) só enxergam a interface
`ProblemaPermutacao`. Nenhum deles importa o problema concreto.

## Passo a passo para reproduzir tudo

| # | Comando | Tempo* | Gera |
|---|---------|--------|------|
| 0 | `python -m unittest discover -s tests -t .` | 2 s | 24 testes |
| 1 | `python main.py verificar` | 1 s | saída da T1 no terminal |
| 2 | `python main.py vizinhancas` | 2 min | `resultados/vizinhancas.csv`, `resultados/avaliacao_incremental.csv` |
| 3 | `python main.py variantes` | 1 min | `resultados/variantes.csv` |
| 4 | `python main.py calibracao-ga` | 2,5 min | `resultados/calibracao_ga.csv` |
| 5 | `python main.py experimentos` | 2,5 min | `resultados/experimentos.csv` |
| 6 | `python main.py ga-hipotese` | 5 min | `resultados/ga_hipotese.csv` |
| 7 | `python main.py analise` | 5 s | tabelas no terminal e `resultados/tabela_*.csv`, `resultados/testes_*.csv` |
| 8 | `python gerar_figuras.py` | 15 s | `Figuras/*.png` |

\* Medido numa máquina com 12 núcleos (11 processos em paralelo).

`python main.py todos` executa os passos 1 a 7 em sequência. A ordem importa: a vizinhança, a variante
do GD e os parâmetros do GA usados nos passos 5 e 6 foram escolhidos a partir dos resultados dos
passos 2 a 4. Eles ficam fixos no início de `services/executor.py` (`VIZINHANCA_PADRAO`,
`VARIANTE_PADRAO`, `PARAMETROS_GA`, `MUTACAO_GA`).

Opções dos experimentos:

- `--rapido`: orçamento/10 e 3 execuções, para conferir o pipeline em segundos. Grava com sufixo
  `_rapido` e não sobrescreve os resultados completos.
- `--processos N`: número de processos em paralelo (padrão: núcleos − 1).
- `--execucoes N`: sobrescreve o número de execuções do protocolo.

Os experimentos são **determinísticos**. Cada execução tem semente própria, derivada de
(experimento, instância, algoritmo, número da execução) por CRC32, então repetir um comando reproduz
os mesmos valores de h. Os tempos de parede variam de uma máquina para outra.

## Onde está cada tabela e figura do relatório

| Item do relatório | Origem |
|---|---|
| Saída da T1 (4366, 3168, 1875, Figura 1) | `python main.py verificar` |
| Tamanho das vizinhanças em função de n | `python main.py verificar` (formulas em `utils/vizinhancas.py`) |
| T2: comparação das vizinhanças + testes | `analise` → `resultados/tabela_vizinhancas.csv`, `testes_vizinhancas.csv` |
| T2: custo por avaliação (completa × incremental) | `resultados/avaliacao_incremental.csv`, `Figuras/avaliacao_incremental.png` |
| Passos de melhoria por variante do GD | `analise` → `resultados/tabela_variantes.csv`, `testes_variantes.csv` |
| Calibração do GA | `analise` → `resultados/tabela_calibracao_ga.csv` |
| Tabela principal (9 combinações, gap na pequena) | `analise` → `resultados/tabela_principal.csv` |
| Testes de Mann–Whitney par a par | `analise` → `resultados/testes_principais.csv` |
| Tempo médio × avaliações | `analise` (bloco "TEMPO x AVALIACOES") → `tabela_principal.csv` |
| Ótimos locais na pequena / execuções que atingem h* | `analise` → `resultados/otimos_locais_pequena.csv`, `Figuras/otimos_locais_pequena.png` |
| Boxplots por instância | `Figuras/boxplots.png` |
| Curvas de convergência (com a média da população do GA) | `Figuras/convergencia.png` |
| Taxa de aceitação de pioras do SA | `Figuras/sa_aceitacao.png` |
| Diversidade da população do GA | `Figuras/ga_diversidade.png` |
| Experimento adicional sobre a recombinação | `analise` → `resultados/tabela_hipotese_ga.csv`, `testes_hipotese_ga.csv`, `Figuras/ga_hipotese.png` |
| Figura da T2 (boxplots por vizinhança) | `Figuras/vizinhancas.png` |

## Decisões que afetam os números

- **Tempo de deslocamento (Eq. 1).** Arredondamento "meio para cima" explícito. Com as coordenadas
  inteiras das instâncias, nenhuma distância termina em .5, então o resultado coincide com `round`.
- **Guloso de referência (3168).** Escolhe o ativo de menor acréscimo imediato de atraso e desempata
  pelo **menor deslocamento** (depois pelo índice). Outras regras de desempate dão de 3815 a 7334.
- **Ótimo exato (1875).** Programação dinâmica sobre (ativos visitados, último ativo), guardando só
  rótulos (instante, custo) não dominados. Executa em ~0,4 s.
- **Contagem de avaliações.** Contada num único lugar (`utils/metricas.py: Monitor`): cada chamada de
  `h(s)` ou `avaliar_sucessor(s, mov)` conta 1, inclusive o h de cada ponto de partida do restart e
  as amostras de calibração do SA. Ao chegar ao orçamento, a próxima avaliação levanta
  `OrcamentoEsgotado`, e toda execução gasta exatamente B avaliações. Duas situações não contam:
  - reconstruir o cache da solução corrente após aceitar um movimento, porque o h dela já foi contado
    quando ela foi examinada;
  - um filho do GA que não sofreu cruzamento nem mutação (é cópia do pai e herda a aptidão dele).
- **Avaliação incremental.** Um movimento nas posições i..j não altera o prefixo antes de min(i, j).
  O sucessor é avaliado a partir dali, com o instante de conclusão e o custo acumulado do prefixo
  guardados em cache. Como h não decresce ao longo do prefixo, a avaliação também pode parar assim
  que o custo parcial atinge um limite (o h corrente na descida, o limiar de aceitação no SA). Isso
  muda o tempo de parede, não a contagem. A concordância com a avaliação completa é verificada em
  `tests/test_incremental.py`.
- **Ordem de varredura.** A lista de movimentos é embaralhada uma vez por execução, e cada varredura
  começa num ponto sorteado. Na ordem lexicográfica, o first improvement testaria seguidamente
  inserções do mesmo ativo, que são fortemente correlacionadas.
- **SA.** Um random walk de 500 passos (contados no orçamento) mede as pioras Δ > 0 na própria
  instância. T0 aceita a piora mediana com probabilidade 0,95; Tf aceita o decil inferior das pioras
  com probabilidade 0,001. α é calculado para T chegar a Tf exatamente no fim do orçamento, com n
  iterações por temperatura. O teste de Gibbs é implementado de forma equivalente: sorteia-se u antes
  e aceita-se s' se h(s') < h(s) − T ln u.
- **GA.** N = 100, torneio k = 3 (h menor vence, sem conversão de aptidão), OX com Pc = 0,6, mutação
  por inserção com Pm = 0,5 e elitismo de 2. Esses valores saíram de `calibracao-ga` (grade de
  18 combinações na instância intermediária, com sementes diferentes das dos experimentos finais).
- **Tempo de parede.** Medido em cada execução com os demais processos rodando em paralelo, o que
  infla os tempos absolutos de forma parecida para todos os algoritmos. A tabela de
  `avaliacao_incremental.csv` roda num único processo. Para tempos sem disputa de CPU, use
  `--processos 1`.
