"""Operadores de vizinhanca sobre permutacoes (Figura 3 do enunciado).

As posicoes sao os indices ``0..n-1`` da lista. Todo movimento altera apenas
as posicoes do intervalo ``[min(i, j), max(i, j)]``: o prefixo anterior fica
intacto, o que permite a avaliacao incremental, e o sufixo posterior mantem
os mesmos ativos na mesma ordem (apenas deslocados no tempo).

Tamanho da vizinhanca induzida (sucessores distintos de uma permutacao):

* ``swap``     -- troca os ativos das posicoes ``i < j``: ``n(n-1)/2``;
* ``insercao`` -- retira o ativo da posicao ``i`` e o reinsere na posicao
  ``j``: ``(n-1)^2``. Dos ``n(n-1)`` pares ordenados, os ``n-1`` pares
  ``j = i - 1`` repetem o sucessor do par ``(j, i)`` e sao descartados;
* ``2opt``     -- inverte o trecho entre as posicoes ``i < j``: ``n(n-1)/2``.
"""

import random
from typing import List, Sequence

from entities.movimento import Movimento

SWAP = "swap"
INSERCAO = "insercao"
DOIS_OPT = "2opt"

VIZINHANCAS = (SWAP, INSERCAO, DOIS_OPT)


def _validar(tipo: str) -> None:
    if tipo not in VIZINHANCAS:
        raise ValueError(f"vizinhanca desconhecida: {tipo!r} (use {VIZINHANCAS})")


def tamanho(tipo: str, n: int) -> int:
    """Numero de sucessores distintos que ``tipo`` induz sobre ``n`` ativos."""
    _validar(tipo)
    if tipo == INSERCAO:
        return (n - 1) ** 2
    return n * (n - 1) // 2


def enumerar(tipo: str, n: int) -> List[Movimento]:
    """Todos os movimentos distintos de ``tipo``, em ordem lexicografica de ``(i, j)``."""
    _validar(tipo)
    if tipo == INSERCAO:
        return [
            Movimento(tipo, i, j)
            for i in range(n)
            for j in range(n)
            if j != i and j != i - 1
        ]
    return [Movimento(tipo, i, j) for i in range(n) for j in range(i + 1, n)]


def sortear(tipo: str, n: int, rng: random.Random) -> Movimento:
    """Sorteia um movimento distinto de ``tipo`` com probabilidade uniforme."""
    _validar(tipo)
    while True:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        if tipo == INSERCAO:
            # Rejeitar j = i - 1 mantem a distribuicao uniforme sobre os
            # (n-1)^2 sucessores distintos: sem isso, as trocas adjacentes
            # teriam o dobro da probabilidade das demais insercoes.
            if j == i - 1:
                continue
            return Movimento(tipo, i, j)
        return Movimento(tipo, i, j) if i < j else Movimento(tipo, j, i)


def trecho(s: Sequence[int], mov: Movimento) -> List[int]:
    """Novo conteudo das posicoes ``min(i, j)..max(i, j)`` apos o movimento."""
    i, j = mov.i, mov.j
    if mov.tipo == SWAP:
        return [s[j]] + list(s[i + 1:j]) + [s[i]]
    if mov.tipo == INSERCAO:
        if i < j:
            return list(s[i + 1:j + 1]) + [s[i]]
        return [s[i]] + list(s[j:i])
    if mov.tipo == DOIS_OPT:
        return list(reversed(s[i:j + 1]))
    raise ValueError(f"vizinhanca desconhecida: {mov.tipo!r}")


def aplicar(s: Sequence[int], mov: Movimento) -> List[int]:
    """Sucessor de ``s`` gerado por ``mov`` (nova lista; ``s`` nao e alterada)."""
    inicio = min(mov.i, mov.j)
    fim = max(mov.i, mov.j)
    return list(s[:inicio]) + trecho(s, mov) + list(s[fim + 1:])
