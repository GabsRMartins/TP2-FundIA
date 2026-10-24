"""Ponto de entrada do TP2 -- Sequenciamento de Manutencao de Ativos.

Compara greedy descent com random restart, simulated annealing e genetic
algorithm sob o mesmo orcamento de avaliacoes da funcao ``h``. Cada
subcomando grava seus resultados em ``resultados/``; ``analise`` le esses
arquivos e imprime/grava as tabelas do relatorio.

Uso::

    python main.py verificar         # T1: valores de verificacao do enunciado
    python main.py vizinhancas       # T2: swap x insercao x 2-opt + custo da avaliacao
    python main.py variantes         # best x first x sucessor aleatorio
    python main.py calibracao-ga     # grade de parametros do GA (instancia media)
    python main.py experimentos      # T3: GD x SA x GA nas tres instancias
    python main.py ga-hipotese       # experimento adicional da Secao 5
    python main.py analise           # tabelas e testes estatisticos
    python main.py todos             # tudo acima, na ordem

Opcoes dos experimentos: ``--rapido`` (orcamento/10 e 3 execucoes, para teste
de fumaca), ``--processos N`` e ``--execucoes N``.
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entities.instancia import carregar_instancia  # noqa: E402
from services import analise  # noqa: E402
from services.executor import (  # noqa: E402
    INSTANCIAS,
    MUTACAO_GA,
    PARAMETROS_GA,
    VARIANTE_PADRAO,
    VIZINHANCA_PADRAO,
    configuracoes_calibracao_ga,
    configuracoes_hipotese_ga,
    configuracoes_principais,
    configuracoes_variantes,
    configuracoes_vizinhancas,
    gravar_tabela,
    instancia,
    medir_avaliacao,
    rodar_experimento,
)
from utils import vizinhancas  # noqa: E402
from utils.referencia import guloso_menor_acrescimo, otimo_exato  # noqa: E402

LARGURA = 78


def verificar() -> None:
    """T1: confere a funcao de avaliacao com os valores do enunciado."""
    print("=" * LARGURA)
    print("T1 - VERIFICACAO DA FUNCAO DE AVALIACAO")
    print("=" * LARGURA)

    exemplo = instancia_exemplo()
    print("Figura 1 (n = 4, base em (0, 0)):")
    for rota, esperado in (([1, 4, 2, 3], 39), ([1, 4, 3, 2], 22)):
        print(f"  h{tuple(rota)} = {exemplo.atraso_ponderado(rota):>4}   (enunciado: {esperado})")
        for ativo, conclusao, atraso in exemplo.cronograma(rota):
            print(f"      ativo {ativo}: C = {conclusao:>2}, w(C - d)+ = {atraso}")
    custo, ordem = min(
        (exemplo.atraso_ponderado(p), p) for p in itertools.permutations(range(1, 5))
    )
    print(f"  forca bruta sobre as 4! permutacoes: otimo h{ordem} = {custo}")

    pequena = instancia("pequena")
    print("\nInstancia pequena (n = 12):")
    identidade = list(range(1, pequena.n + 1))
    print(f"  identidade (1, ..., 12)            : h = {pequena.atraso_ponderado(identidade)}"
          "   (enunciado: 4366)")
    custo, ordem = guloso_menor_acrescimo(pequena)
    print(f"  guloso de menor acrescimo de atraso: h = {custo}   (enunciado: 3168)")
    print(f"      ordem: {ordem}")
    inicio = time.perf_counter()
    custo, ordem = otimo_exato(pequena)
    print(f"  otimo exato (programacao dinamica) : h = {custo}   (enunciado: 1875)"
          f"   [{time.perf_counter() - inicio:.2f} s]")
    print(f"      ordem: {ordem}")

    print("\nTamanho da vizinhanca induzida por operador:")
    print(f"  {'n':>5}" + "".join(f"{tipo:>12}" for tipo in vizinhancas.VIZINHANCAS))
    for nome in INSTANCIAS:
        n = instancia(nome).n
        print(f"  {n:>5}" + "".join(
            f"{vizinhancas.tamanho(tipo, n):>12}" for tipo in vizinhancas.VIZINHANCAS))
    print("  formulas: swap = n(n-1)/2, insercao = (n-1)^2, 2opt = n(n-1)/2")
    print("=" * LARGURA)


def instancia_exemplo():
    raiz = os.path.dirname(os.path.abspath(__file__))
    return carregar_instancia(os.path.join(raiz, "manutencao_exemplo.txt"))


def cmd_vizinhancas(args) -> None:
    rodar_experimento("vizinhancas", configuracoes_vizinhancas(), **_opcoes(args))
    print("\nCusto por avaliacao (processo unico):")
    linhas = medir_avaliacao(amostras=2_000 if args.rapido else 20_000)
    gravar_tabela("avaliacao_incremental" + ("_rapido" if args.rapido else ""), linhas)


def cmd_variantes(args) -> None:
    rodar_experimento("variantes", configuracoes_variantes(), **_opcoes(args))


def cmd_calibracao_ga(args) -> None:
    opcoes = _opcoes(args)
    opcoes["execucoes"] = opcoes["execucoes"] or 10
    rodar_experimento("calibracao_ga", configuracoes_calibracao_ga(), instancias=("media",),
                      **opcoes)


def cmd_experimentos(args) -> None:
    print(f"GD: variante {VARIANTE_PADRAO}, vizinhanca {VIZINHANCA_PADRAO}")
    print(f"GA: {PARAMETROS_GA}, mutacao {MUTACAO_GA}")
    rodar_experimento("experimentos", configuracoes_principais(), **_opcoes(args))


def cmd_ga_hipotese(args) -> None:
    rodar_experimento("ga_hipotese", configuracoes_hipotese_ga(), **_opcoes(args))


def _opcoes(args):
    return {"rapido": args.rapido, "processos": args.processos, "execucoes": args.execucoes}


COMANDOS = {
    "verificar": lambda args: verificar(),
    "vizinhancas": cmd_vizinhancas,
    "variantes": cmd_variantes,
    "calibracao-ga": cmd_calibracao_ga,
    "experimentos": cmd_experimentos,
    "ga-hipotese": cmd_ga_hipotese,
    "analise": lambda args: analise.analisar_tudo(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="TP2 - Busca Local e Algoritmo Genetico")
    parser.add_argument("comando", choices=[*COMANDOS, "todos"])
    parser.add_argument("--rapido", action="store_true",
                        help="orcamento/10 e 3 execucoes (teste de fumaca)")
    parser.add_argument("--processos", type=int, default=None,
                        help="processos em paralelo (padrao: nucleos - 1)")
    parser.add_argument("--execucoes", type=int, default=None,
                        help="sobrescreve o numero de execucoes do protocolo")
    args = parser.parse_args()

    if args.comando == "todos":
        for nome, comando in COMANDOS.items():
            if args.rapido and nome == "analise":
                continue  # a analise le os arquivos completos
            comando(args)
    else:
        COMANDOS[args.comando](args)


if __name__ == "__main__":
    main()
