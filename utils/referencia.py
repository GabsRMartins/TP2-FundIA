"""Solucoes de referencia usadas para validar a funcao de avaliacao.

Nenhuma das duas faz parte dos algoritmos comparados: o guloso reproduz o
valor de calibracao do enunciado (3168 na instancia pequena) e a programacao
dinamica e uma busca exata independente que confirma o otimo global (1875).
"""

from typing import Dict, List, Tuple

from entities.instancia import Instancia


def guloso_menor_acrescimo(instancia: Instancia) -> Tuple[int, List[int]]:
    """Visita sempre o ativo que gera o menor acrescimo imediato de atraso.

    Empates (comuns no inicio, quando varios ativos ainda nao atrasam) sao
    desfeitos pelo menor deslocamento a partir da posicao atual e, depois,
    pelo menor indice. Essa regra reproduz os 3168 do enunciado; desempatar
    pelo indice, pelo prazo ou pelo instante de conclusao da 3815 a 7334.
    """
    t, p, d, w = instancia.t, instancia.p, instancia.d, instancia.w
    restantes = set(range(1, instancia.n + 1))
    conclusao = 0
    anterior = 0
    total = 0
    ordem: List[int] = []
    while restantes:
        def chave(j: int) -> Tuple[int, int, int]:
            fim = conclusao + t[anterior][j] + p[j]
            return (w[j] * max(0, fim - d[j]), t[anterior][j], j)

        escolhido = min(restantes, key=chave)
        conclusao += t[anterior][escolhido] + p[escolhido]
        total += w[escolhido] * max(0, conclusao - d[escolhido])
        ordem.append(escolhido)
        restantes.remove(escolhido)
        anterior = escolhido
    return total, ordem


def otimo_exato(instancia: Instancia) -> Tuple[int, List[int]]:
    """Otimo global por programacao dinamica sobre ``(visitados, ultimo)``.

    O custo futuro depende apenas do conjunto ainda nao visitado, do ultimo
    ativo visitado e do instante atual, e nao diminui quando esse instante
    aumenta. Por isso, entre dois rotulos ``(instante, custo)`` do mesmo par
    ``(visitados, ultimo)``, o que for pior ou igual nos dois criterios pode
    ser descartado sem perder o otimo: cada estado guarda so a fronteira de
    Pareto. Exponencial em ``n`` (``2^n * n`` estados), viavel para n = 12.
    """
    n = instancia.n
    if n > 16:
        raise ValueError("busca exata so e viavel para instancias pequenas")
    t, p, d, w = instancia.t, instancia.p, instancia.d, instancia.w

    # rotulo: (instante, custo, ordem parcial)
    Rotulo = Tuple[int, int, Tuple[int, ...]]
    camada: Dict[Tuple[int, int], List[Rotulo]] = {(0, 0): [(0, 0, ())]}
    for _ in range(n):
        proxima: Dict[Tuple[int, int], List[Rotulo]] = {}
        for (visitados, ultimo), rotulos in camada.items():
            for j in range(1, n + 1):
                if visitados >> j & 1:
                    continue
                destino = proxima.setdefault((visitados | 1 << j, j), [])
                for instante, custo, ordem in rotulos:
                    fim = instante + t[ultimo][j] + p[j]
                    destino.append((fim, custo + w[j] * max(0, fim - d[j]), ordem + (j,)))
        camada = {chave: _fronteira_pareto(rotulos) for chave, rotulos in proxima.items()}

    custo, ordem = min((r[1], r[2]) for rotulos in camada.values() for r in rotulos)
    return custo, list(ordem)


def _fronteira_pareto(rotulos):
    """Rotulos nao dominados em ``(instante, custo)``, ambos a minimizar."""
    rotulos.sort()
    fronteira = []
    melhor_custo = None
    for rotulo in rotulos:
        if melhor_custo is None or rotulo[1] < melhor_custo:
            fronteira.append(rotulo)
            melhor_custo = rotulo[1]
    return fronteira
