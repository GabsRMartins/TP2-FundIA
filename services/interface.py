"""Interface comum entre os algoritmos e o problema (Secao 2.1 do enunciado).

Os tres algoritmos conhecem apenas esta interface. Nada neles depende de
ativos, prazos ou deslocamentos: trocar o problema exige apenas outra
implementacao de :class:`ProblemaPermutacao`.

Um sucessor e descrito pelo :class:`Movimento` que o gera. ``sucessores`` e
``sucessor_aleatorio`` devolvem movimentos; ``aplicar`` materializa a
permutacao e ``avaliar_sucessor`` calcula ``h`` do sucessor sem materializa-lo.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Sequence

from entities.movimento import Movimento
from utils.metricas import INFINITO, Monitor


class ProblemaPermutacao(ABC):
    """Problema de minimizacao cujas solucoes candidatas sao permutacoes."""

    #: Toda avaliacao de ``h`` passa por este monitor.
    monitor: Monitor

    @abstractmethod
    def solucao_aleatoria(self) -> List[int]:
        """Gera um ponto de partida (uma permutacao sorteada uniformemente)."""

    @abstractmethod
    def h(self, s: Sequence[int]) -> int:
        """Funcao de avaliacao, sempre de minimizacao. Conta 1 avaliacao."""

    @abstractmethod
    def sucessores(self, s: Sequence[int]) -> Iterator[Movimento]:
        """Itera sobre todos os sucessores de ``s``."""

    @abstractmethod
    def sucessor_aleatorio(self, s: Sequence[int]) -> Movimento:
        """Sorteia um unico sucessor de ``s``."""

    @abstractmethod
    def tamanho_vizinhanca(self) -> int:
        """Numero de sucessores de cada solucao."""

    @abstractmethod
    def avaliar_sucessor(
        self, s: Sequence[int], mov: Movimento, limite: float = INFINITO
    ) -> int:
        """``h`` do sucessor de ``s`` gerado por ``mov``. Conta 1 avaliacao.

        Se o valor for menor que ``limite`` ele e exato. Caso contrario, a
        avaliacao pode ser interrompida e o retorno e apenas um limitante
        inferior que ja atinge ``limite`` -- suficiente para quem so precisa
        saber se o sucessor fica abaixo de ``limite``.
        """

    @abstractmethod
    def aplicar(self, s: Sequence[int], mov: Movimento) -> List[int]:
        """Sucessor de ``s`` gerado por ``mov``."""

    @abstractmethod
    def cruzamento(self, s1: Sequence[int], s2: Sequence[int]) -> List[int]:
        """Operador de recombinacao."""

    @abstractmethod
    def mutacao(self, s: Sequence[int]) -> List[int]:
        """Operador de mutacao."""
