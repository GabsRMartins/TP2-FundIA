"""Movimento que leva uma permutacao a um de seus sucessores."""

from typing import NamedTuple


class Movimento(NamedTuple):
    """Movimento ``(tipo, i, j)`` sobre as posicoes ``i`` e ``j`` da permutacao.

    Um sucessor e representado pelo movimento que o gera, e nao pela
    permutacao ja materializada: assim a vizinhanca inteira pode ser
    percorrida sem copiar ``n`` listas de tamanho ``n``, e a avaliacao
    incremental sabe exatamente quais posicoes mudaram. ``NamedTuple`` em vez
    de ``dataclass`` porque sao criados centenas de milhares de movimentos por
    execucao e a tupla e mais barata de construir.
    """

    tipo: str
    i: int
    j: int
