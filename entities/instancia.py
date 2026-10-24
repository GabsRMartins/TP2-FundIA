"""Instancia do problema: base, ativos e matriz de tempos de deslocamento."""

import math
import os
from typing import List, Sequence, Tuple

from .ativo import Ativo


def tempo_deslocamento(xa: float, ya: float, xb: float, yb: float) -> int:
    """Eq. (1): distancia euclidiana arredondada para o inteiro mais proximo.

    Usa o arredondamento "meio para cima" explicito em vez de ``round``, que
    em Python arredonda ``x.5`` para o par. Com coordenadas inteiras a
    distancia nunca termina exatamente em ``.5``, mas a regra fica
    independente disso.
    """
    return int(math.floor(math.hypot(xa - xb, ya - yb) + 0.5))


class Instancia:
    """Dados de uma instancia, indexados de forma que o ponto 0 e a base.

    ``t[i][j]``: tempo de deslocamento entre os pontos ``i`` e ``j``.
    ``p``, ``d``, ``w``: listas de tamanho ``n + 1`` com ``p[0] = d[0] = w[0] = 0``.
    """

    def __init__(self, nome: str, base: Tuple[float, float], ativos: Sequence[Ativo]) -> None:
        self.nome = nome
        self.base = base
        self.ativos: Tuple[Ativo, ...] = tuple(ativos)
        self.n = len(self.ativos)

        pontos = [base] + [(a.x, a.y) for a in self.ativos]
        self.t: List[List[int]] = [
            [tempo_deslocamento(xi, yi, xj, yj) for xj, yj in pontos] for xi, yi in pontos
        ]
        self.p: List[int] = [0] + [a.p for a in self.ativos]
        self.d: List[int] = [0] + [a.d for a in self.ativos]
        self.w: List[int] = [0] + [a.w for a in self.ativos]

    def atraso_ponderado(self, perm: Sequence[int]) -> int:
        """Funcao de avaliacao ``h(pi)`` das Eqs. (2) e (3), calculada do zero.

        E a implementacao de referencia: nao conta avaliacoes e serve para
        validar a avaliacao (incremental) usada pelos algoritmos.
        """
        t, p, d, w = self.t, self.p, self.d, self.w
        conclusao = 0
        anterior = 0
        total = 0
        for j in perm:
            conclusao += t[anterior][j] + p[j]
            if conclusao > d[j]:
                total += w[j] * (conclusao - d[j])
            anterior = j
        return total

    def cronograma(self, perm: Sequence[int]) -> List[Tuple[int, int, int]]:
        """Lista ``(ativo, C_j, w_j * max(0, C_j - d_j))`` na ordem de visita."""
        t, p, d, w = self.t, self.p, self.d, self.w
        linhas = []
        conclusao = 0
        anterior = 0
        for j in perm:
            conclusao += t[anterior][j] + p[j]
            linhas.append((j, conclusao, w[j] * max(0, conclusao - d[j])))
            anterior = j
        return linhas

    def __repr__(self) -> str:
        return f"Instancia({self.nome!r}, n={self.n})"


def carregar_instancia(caminho: str) -> Instancia:
    """Le um arquivo ``manutencao_*.txt`` no formato da Secao 1.3."""
    with open(caminho, encoding="utf-8") as arquivo:
        valores = arquivo.read().split()

    n = int(valores[0])
    if len(valores) != 3 + 5 * n:
        raise ValueError(f"{caminho}: esperados {n} ativos com 5 campos cada")
    base = (float(valores[1]), float(valores[2]))
    ativos = []
    for k in range(n):
        x, y, p, d, w = valores[3 + 5 * k: 8 + 5 * k]
        ativos.append(Ativo(indice=k + 1, x=float(x), y=float(y), p=int(p), d=int(d), w=int(w)))

    nome = os.path.splitext(os.path.basename(caminho))[0].replace("manutencao_", "")
    return Instancia(nome, base, ativos)
