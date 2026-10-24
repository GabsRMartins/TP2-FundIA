"""Operadores de recombinacao que preservam a validade da permutacao.

Os cruzamentos de um ou varios pontos copiariam ativos repetidos de cada
pai; os dois operadores abaixo garantem que o filho visite cada ativo
exatamente uma vez. Os pontos de corte ``a <= b`` sao posicoes ``0..n-1``,
inclusivas.
"""

import random
from typing import List, Sequence, Tuple


def sortear_cortes(n: int, rng: random.Random) -> Tuple[int, int]:
    """Dois pontos de corte ``a <= b`` sorteados uniformemente."""
    a = rng.randrange(n)
    b = rng.randrange(n)
    return (a, b) if a <= b else (b, a)


def ox(pai1: Sequence[int], pai2: Sequence[int], a: int, b: int) -> List[int]:
    """Order Crossover (OX).

    O filho herda de ``pai1`` o trecho ``a..b`` na mesma posicao. As demais
    posicoes, a partir de ``b + 1`` e voltando ao inicio, recebem os ativos
    que faltam na ordem relativa em que aparecem em ``pai2``, tambem lida a
    partir de ``b + 1``.
    """
    n = len(pai1)
    filho: List[int] = [0] * n
    filho[a:b + 1] = pai1[a:b + 1]
    herdados = set(pai1[a:b + 1])

    posicao = (b + 1) % n
    for k in range(n):
        ativo = pai2[(b + 1 + k) % n]
        if ativo in herdados:
            continue
        filho[posicao] = ativo
        posicao = (posicao + 1) % n
    return filho


def pmx(pai1: Sequence[int], pai2: Sequence[int], a: int, b: int) -> List[int]:
    """Partially Mapped Crossover (PMX).

    O filho herda de ``pai1`` o trecho ``a..b``. Cada ativo de ``pai2`` desse
    trecho que ficou de fora e colocado seguindo o mapeamento
    ``pai1[k] -> pai2[k]`` ate cair fora do trecho; as posicoes restantes sao
    copiadas de ``pai2``. Ao contrario do OX, preserva posicoes absolutas.
    """
    n = len(pai1)
    filho: List[int] = [0] * n
    preenchido = [False] * n
    for k in range(a, b + 1):
        filho[k] = pai1[k]
        preenchido[k] = True
    posicao_em_pai2 = {ativo: k for k, ativo in enumerate(pai2)}
    herdados = set(pai1[a:b + 1])

    for k in range(a, b + 1):
        ativo = pai2[k]
        if ativo in herdados:
            continue
        destino = k
        while a <= destino <= b:
            destino = posicao_em_pai2[pai1[destino]]
        filho[destino] = ativo
        preenchido[destino] = True

    for k in range(n):
        if not preenchido[k]:
            filho[k] = pai2[k]
    return filho
