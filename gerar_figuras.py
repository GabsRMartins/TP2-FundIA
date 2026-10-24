"""Geracao das figuras do relatorio do TP2 a partir de ``resultados/``.

Nao executa nenhum algoritmo: le os CSV/JSON gravados por ``main.py`` e
produz em ``Figuras/``:

* ``boxplots.png``              -- melhor h por execucao, GD x SA x GA, por instancia;
* ``convergencia.png``          -- mediana do melhor h x avaliacoes (e media da populacao do GA);
* ``sa_aceitacao.png``          -- taxa de aceitacao de pioras e temperatura do SA;
* ``ga_diversidade.png``        -- diversidade da populacao do GA ao longo das avaliacoes;
* ``otimos_locais_pequena.png`` -- frequencia de cada otimo local do GD na instancia pequena;
* ``vizinhancas.png``           -- T2: GD (first improvement) com swap, insercao e 2-opt;
* ``avaliacao_incremental.png`` -- custo por avaliacao: completa x incremental x com corte;
* ``ga_hipotese.png``           -- experimento adicional sobre a recombinacao.

Uso::

    python gerar_figuras.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors, ticker  # noqa: E402

from services.executor import INSTANCIAS, ler_csv, ler_tracos  # noqa: E402
from services.analise import agrupar, existe  # noqa: E402

DESTINO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figuras")

# Paleta de referencia validada (skill de visualizacao): os tres primeiros
# slots categoricos passam em todas as comparacoes par a par, inclusive sob
# daltonismo. A cor segue a entidade: GD, SA e GA tem sempre a mesma cor.
CORES = {"GD": "#2a78d6", "SA": "#eb6834", "GA": "#1baf7a"}
RAMPA_AZUL = ["#86b6ef", "#3987e5", "#184f95"]  # ordinal, do mais claro ao mais escuro
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
SUPERFICIE = "#ffffff"

NOMES_VIZINHANCAS = {"swap": "swap", "insercao": "inserção", "2opt": "2-opt"}
TITULOS = {"pequena": "pequena (n = 12)", "media": "intermediária (n = 50)",
           "grande": "grande (n = 150)"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": EIXO,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "axes.titlecolor": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.6,
    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "xtick.labelcolor": TINTA_SECUNDARIA,
    "ytick.labelcolor": TINTA_SECUNDARIA,
    "legend.frameon": False,
    "legend.labelcolor": TINTA_SECUNDARIA,
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "lines.linewidth": 1.6,
})


def _salvar(fig, nome: str) -> None:
    for ax in fig.axes:
        for eixo in (ax.xaxis, ax.yaxis):
            if type(eixo.get_major_formatter()) is ticker.ScalarFormatter:
                eixo.set_major_formatter(ticker.FuncFormatter(_decimal))
    os.makedirs(DESTINO, exist_ok=True)
    caminho = os.path.join(DESTINO, nome)
    fig.savefig(caminho, dpi=200, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)
    print(f"gerado: {caminho}")


def _decimal(valor: float, _=None) -> str:
    """Numero com virgula decimal; milhares abreviados como "mil"."""
    if abs(valor) >= 1000:
        return f"{valor / 1000:g} mil".replace(".", ",")
    return f"{valor:g}".replace(".", ",")


def _milhares(ax, eixo: str = "y") -> None:
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(ticker.FuncFormatter(_decimal))


def _titulo(ax, nome: str) -> None:
    """Titulo do painel com o valor de referencia da instancia (Tabela 1)."""
    valor = INSTANCIAS[nome].referencia
    rotulo = "ótimo h* = 1875" if nome == "pequena" else f"referência {valor:,}".replace(",", " ")
    ax.set_title(f"{TITULOS[nome]}\n", fontsize=10)
    ax.annotate(rotulo + "  (linha tracejada)", xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=7.5, color=TINTA_SECUNDARIA)


def _referencia(ax, nome: str) -> None:
    ax.axhline(INSTANCIAS[nome].referencia, color=TINTA_SUAVE, linewidth=1,
               linestyle=(0, (4, 3)), zorder=1)
    _titulo(ax, nome)


def _cor(rotulo: str) -> str:
    for chave, cor in CORES.items():
        if rotulo.startswith(chave):
            return cor
    return CORES["GD"]


def _boxplot(ax, grupos, rotulos, cores) -> None:
    """Caixas finas com os pontos de cada execucao sobrepostos."""
    posicoes = np.arange(1, len(grupos) + 1)
    caixas = ax.boxplot(
        grupos, positions=posicoes, widths=0.45, patch_artist=True, showfliers=False,
        medianprops={"color": TINTA, "linewidth": 1.4},
        whiskerprops={"color": TINTA_SUAVE, "linewidth": 1},
        capprops={"color": TINTA_SUAVE, "linewidth": 1},
        boxprops={"linewidth": 1},
    )
    for caixa, cor in zip(caixas["boxes"], cores):
        caixa.set_facecolor(mcolors.to_rgba(cor, 0.18))
        caixa.set_edgecolor(cor)
    rng = np.random.default_rng(0)
    for posicao, valores, cor in zip(posicoes, grupos, cores):
        deslocamento = rng.uniform(-0.12, 0.12, len(valores))
        ax.scatter(posicao + deslocamento, valores, s=12, color=cor, alpha=0.75,
                   edgecolors=SUPERFICIE, linewidths=0.5, zorder=3)
    ax.set_xticks(posicoes, rotulos)
    ax.grid(axis="x", visible=False)


def _paineis(colunas: int = 3, linhas: int = 1, altura: float = 2.8, **kwargs):
    fig, eixos = plt.subplots(linhas, colunas, figsize=(3.3 * colunas, altura * linhas),
                              squeeze=False, **kwargs)
    return fig, eixos


# ----------------------------------------------------------------------
# Experimento principal
# ----------------------------------------------------------------------
def figura_boxplots() -> None:
    grupos = agrupar(ler_csv("experimentos"))
    fig, eixos = _paineis()
    for ax, nome in zip(eixos[0], INSTANCIAS):
        por_rotulo = grupos[nome]
        rotulos = list(por_rotulo)
        valores = [[float(l["melhor_h"]) for l in por_rotulo[r]] for r in rotulos]
        _boxplot(ax, valores, rotulos, [_cor(r) for r in rotulos])
        _referencia(ax, nome)
        _milhares(ax)
    eixos[0][0].set_ylabel("melhor h por execução")
    fig.tight_layout()
    _salvar(fig, "boxplots.png")


def _escada(curva, grade):
    """Valor de uma curva em escada ``[(x, y), ...]`` em cada ponto da grade."""
    if not curva:
        return np.full(len(grade), np.nan)
    xs = np.array([p[0] for p in curva], dtype=float)
    ys = np.array([p[1] for p in curva], dtype=float)
    indices = np.searchsorted(xs, grade, side="right") - 1
    valores = ys[np.clip(indices, 0, None)]
    valores[indices < 0] = np.nan
    return valores


def figura_convergencia() -> None:
    tracos = ler_tracos("experimentos")
    fig, eixos = _paineis(altura=3.0)
    for ax, nome in zip(eixos[0], INSTANCIAS):
        orcamento = INSTANCIAS[nome].orcamento
        grade = np.linspace(1, orcamento, 500)
        for rotulo in ("GD", "SA", "GA"):
            execucoes = tracos[f"{nome}|{rotulo}"]
            melhores = np.array([_escada(e["curva"], grade) for e in execucoes])
            ax.plot(grade, np.nanmedian(melhores, axis=0), color=CORES[rotulo],
                    label=f"{rotulo} (melhor)")
            if rotulo == "GA":
                medias = np.array([
                    _escada(list(zip(e["avaliacoes"], e["media"])), grade) for e in execucoes
                ])
                ax.plot(grade, _nanmediana(medias), color=CORES["GA"],
                        linestyle=(0, (3, 2)), linewidth=1.3, label="GA (média da população)")
        _referencia(ax, nome)
        ax.set_yscale("log")
        ax.set_xlabel("avaliações de h")
        _milhares(ax, "x")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_decimal))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    eixos[0][0].set_ylabel("h (mediana das execuções, escala log)")
    manipuladores, rotulos = eixos[0][0].get_legend_handles_labels()
    fig.legend(manipuladores, rotulos, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _salvar(fig, "convergencia.png")


def _nanmediana(matriz):
    """Mediana por coluna; colunas sem nenhum valor (antes da 1a geracao) viram nan."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(matriz, axis=0)


def _mediana_e_faixa(series):
    matriz = np.array(series, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmedian(matriz, axis=0), np.nanpercentile(matriz, 25, axis=0),
                np.nanpercentile(matriz, 75, axis=0))


def figura_aceitacao_sa() -> None:
    tracos = ler_tracos("experimentos")
    fig, eixos = _paineis(linhas=2, altura=2.4, sharex=True)
    for coluna, nome in enumerate(INSTANCIAS):
        execucoes = tracos[f"{nome}|SA"]
        comprimento = min(len(e["avaliacoes"]) for e in execucoes)
        x = np.array(execucoes[0]["avaliacoes"][:comprimento]) / INSTANCIAS[nome].orcamento
        for linha, (chave, rotulo) in enumerate((
            ("taxa_aceitacao_pioras", "aceitação de pioras"),
            ("temperatura", "temperatura T (log)"),
        )):
            ax = eixos[linha][coluna]
            mediana, q1, q3 = _mediana_e_faixa([e[chave][:comprimento] for e in execucoes])
            ax.fill_between(x, q1, q3, color=CORES["SA"], alpha=0.18, linewidth=0)
            ax.plot(x, mediana, color=CORES["SA"])
            if chave == "temperatura":
                ax.set_yscale("log")
                ax.set_xlabel("fração do orçamento")
            else:
                ax.set_ylim(-0.02, 1.02)
                ax.set_title(TITULOS[nome])
            if coluna == 0:
                ax.set_ylabel(rotulo)
    eixos[0][0].annotate("mediana; faixa = quartis", xy=(0.98, 0.95), xycoords="axes fraction",
                         ha="right", va="top", fontsize=7.5, color=TINTA_SECUNDARIA)
    fig.tight_layout()
    _salvar(fig, "sa_aceitacao.png")


def figura_diversidade_ga() -> None:
    tracos = ler_tracos("experimentos")
    fig, eixos = _paineis(linhas=2, altura=2.4, sharex="col")
    for coluna, nome in enumerate(INSTANCIAS):
        execucoes = tracos[f"{nome}|GA"]
        grade = np.linspace(1, INSTANCIAS[nome].orcamento, 400)
        for linha, (chave, rotulo) in enumerate((
            ("hamming", "Hamming posicional média"),
            ("distintos", "cromossomos distintos"),
        )):
            ax = eixos[linha][coluna]
            series = [_escada(list(zip(e["avaliacoes"], e[chave])), grade) for e in execucoes]
            mediana, q1, q3 = _mediana_e_faixa(series)
            ax.fill_between(grade, q1, q3, color=CORES["GA"], alpha=0.18, linewidth=0)
            ax.plot(grade, mediana, color=CORES["GA"])
            if linha == 0:
                ax.set_ylim(-0.02, 1.02)
                ax.set_title(TITULOS[nome])
            else:
                ax.set_ylim(0, 105)
                ax.set_xlabel("avaliações de h")
                _milhares(ax, "x")
            if coluna == 0:
                ax.set_ylabel(rotulo)
    fig.tight_layout()
    _salvar(fig, "ga_diversidade.png")


def figura_otimos_locais() -> None:
    tracos = ler_tracos("experimentos")
    valores = [v for e in tracos["pequena|GD"] for v in e["valores_otimos"]]
    frequencia = Counter(valores)
    ordenados = sorted(frequencia)
    total = len(valores)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    cores = [CORES["SA"] if v == INSTANCIAS["pequena"].referencia else CORES["GD"]
             for v in ordenados]
    barras = ax.bar(range(len(ordenados)), [100 * frequencia[v] / total for v in ordenados],
                    color=cores, width=0.7, edgecolor=SUPERFICIE, linewidth=1)
    ax.set_xticks(range(len(ordenados)), [str(v) for v in ordenados], rotation=60, fontsize=7.5)
    ax.set_xlabel("valor de h do ótimo local (vizinhança de inserção)")
    ax.set_ylabel("% das descidas")
    ax.grid(axis="x", visible=False)
    otimo = ordenados.index(INSTANCIAS["pequena"].referencia)
    percentual = f"{barras[otimo].get_height():.1f}".replace(".", ",")
    ax.annotate(f"em laranja: ótimo global h* = 1875 ({percentual}% das descidas)",
                xy=(0.99, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.5, color=TINTA_SECUNDARIA)
    ax.set_title(f"Ótimos locais do GD na instância pequena ({total} descidas, 30 execuções)")
    fig.tight_layout()
    _salvar(fig, "otimos_locais_pequena.png")


# ----------------------------------------------------------------------
# T2 e experimentos auxiliares
# ----------------------------------------------------------------------
def figura_vizinhancas() -> None:
    grupos = agrupar(ler_csv("vizinhancas"))
    fig, eixos = _paineis()
    for ax, nome in zip(eixos[0], INSTANCIAS):
        por_rotulo = grupos[nome]
        rotulos = list(por_rotulo)
        valores = [[float(l["melhor_h"]) for l in por_rotulo[r]] for r in rotulos]
        _boxplot(ax, valores, [NOMES_VIZINHANCAS[r.split("/")[1]] for r in rotulos],
                 [CORES["GD"]] * len(rotulos))
        _referencia(ax, nome)
        _milhares(ax)
    eixos[0][0].set_ylabel("melhor h (GD first improvement)")
    fig.tight_layout()
    _salvar(fig, "vizinhancas.png")


def figura_avaliacao_incremental() -> None:
    linhas = [l for l in ler_csv("avaliacao_incremental") if l["contexto"] == "otimo_local"]
    modos = [("us_completa", "completa"), ("us_incremental", "incremental"),
             ("us_incremental_corte", "incremental com corte")]
    fig, eixos = _paineis(altura=2.6)
    for ax, nome in zip(eixos[0], INSTANCIAS):
        dados = [l for l in linhas if l["instancia"] == nome]
        x = np.arange(len(dados))
        largura = 0.26
        for k, ((coluna, rotulo), cor) in enumerate(zip(modos, RAMPA_AZUL)):
            ax.bar(x + (k - 1) * largura, [float(l[coluna]) for l in dados], width=largura - 0.03,
                   color=cor, label=rotulo)
        ax.set_xticks(x, [NOMES_VIZINHANCAS[l["vizinhanca"]] for l in dados])
        ax.grid(axis="x", visible=False)
        ax.set_title(TITULOS[nome])
    eixos[0][0].set_ylabel("microssegundos por avaliação")
    manipuladores, rotulos = eixos[0][0].get_legend_handles_labels()
    fig.legend(manipuladores, rotulos, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _salvar(fig, "avaliacao_incremental.png")


def figura_hipotese_ga() -> None:
    grupos = agrupar(ler_csv("ga_hipotese"))
    nomes_curtos = {"GA-OX": "OX", "GA-PMX": "PMX", "GA-OX-aleatorio": "OX com\npai aleat.",
                    "GA-sem-cruzamento": "só\nmutação"}
    fig, eixos = _paineis(altura=3.0)
    for ax, nome in zip(eixos[0], INSTANCIAS):
        por_rotulo = grupos[nome]
        rotulos = list(por_rotulo)
        valores = [[float(l["melhor_h"]) for l in por_rotulo[r]] for r in rotulos]
        _boxplot(ax, valores, [nomes_curtos.get(r, r) for r in rotulos],
                 [CORES["GA"]] * len(rotulos))
        _referencia(ax, nome)
        ax.tick_params(axis="x", labelsize=7.5)
        _milhares(ax)
    eixos[0][0].set_ylabel("melhor h por execução (GA)")
    fig.tight_layout()
    _salvar(fig, "ga_hipotese.png")


FIGURAS = [
    ("experimentos", figura_boxplots),
    ("experimentos", figura_convergencia),
    ("experimentos", figura_aceitacao_sa),
    ("experimentos", figura_diversidade_ga),
    ("experimentos", figura_otimos_locais),
    ("vizinhancas", figura_vizinhancas),
    ("avaliacao_incremental", figura_avaliacao_incremental),
    ("ga_hipotese", figura_hipotese_ga),
]


def main() -> None:
    for fonte, funcao in FIGURAS:
        if existe(fonte):
            funcao()
        else:
            print(f"pulado: {funcao.__name__} (sem resultados/{fonte}.csv)")


if __name__ == "__main__":
    main()
