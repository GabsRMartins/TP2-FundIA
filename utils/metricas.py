"""Contagem de avaliacoes e registro do desempenho de uma execucao."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

INFINITO = float("inf")


class OrcamentoEsgotado(Exception):
    """Sinaliza que o numero maximo de avaliacoes de ``h`` foi atingido."""


class Monitor:
    """Unico ponto do codigo em que avaliacoes de ``h`` sao contadas.

    O problema chama :meth:`consumir` antes de cada avaliacao e
    :meth:`observar` com o valor exato obtido. Ao atingir o orcamento, a
    proxima tentativa de avaliacao levanta :class:`OrcamentoEsgotado`, de modo
    que toda execucao gasta *exatamente* ``orcamento`` avaliacoes. O monitor
    tambem guarda a melhor solucao vista e a curva de convergencia
    ``(avaliacoes gastas, melhor h ate entao)``, registrada a cada melhora.
    """

    def __init__(self, orcamento: Union[int, float] = INFINITO) -> None:
        self.orcamento = orcamento
        self.avaliacoes = 0
        self.melhor_h: float = INFINITO
        self.melhor_solucao: Optional[List[int]] = None
        self.curva: List[Tuple[int, int]] = []

    @property
    def restante(self) -> Union[int, float]:
        return self.orcamento - self.avaliacoes

    def consumir(self) -> None:
        """Debita uma avaliacao; levanta excecao se o orcamento acabou."""
        if self.avaliacoes >= self.orcamento:
            raise OrcamentoEsgotado
        self.avaliacoes += 1

    def observar(
        self,
        valor: int,
        solucao: Union[Sequence[int], Callable[[], Sequence[int]]],
    ) -> None:
        """Registra o valor exato de uma avaliacao ja debitada.

        ``solucao`` pode ser a propria permutacao ou uma funcao que a constroi;
        a segunda forma evita materializar sucessores que nao melhoram nada.
        """
        if valor < self.melhor_h:
            self.melhor_h = valor
            self.melhor_solucao = list(solucao() if callable(solucao) else solucao)
            self.curva.append((self.avaliacoes, valor))


@dataclass
class ResultadoExecucao:
    """Resultado de uma execucao independente de um algoritmo."""

    algoritmo: str
    melhor_h: int
    melhor_solucao: List[int]
    avaliacoes: int
    curva: List[Tuple[int, int]]
    tempo: float = 0.0
    #: Indicadores escalares especificos do algoritmo (passos, restarts...).
    extras: Dict[str, Any] = field(default_factory=dict)
    #: Series temporais especificas do algoritmo (aceitacao, diversidade...).
    tracos: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def do_monitor(cls, algoritmo: str, monitor: Monitor, **kwargs: Any) -> "ResultadoExecucao":
        return cls(
            algoritmo=algoritmo,
            melhor_h=int(monitor.melhor_h),
            melhor_solucao=list(monitor.melhor_solucao or []),
            avaliacoes=monitor.avaliacoes,
            curva=list(monitor.curva),
            **kwargs,
        )
