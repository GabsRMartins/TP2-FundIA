from dataclasses import dataclass


@dataclass(frozen=True)
class Ativo:
    """Ativo da rede a receber manutencao preventiva.

    ``indice``: identificador ``j`` no arquivo da instancia (1..n).
    ``x``, ``y``: posicao na area de concessao.
    ``p``: duracao do servico, em horas.
    ``d``: prazo regulatorio (hora-limite para a conclusao).
    ``w``: peso de criticidade, o custo por hora de atraso.
    """

    indice: int
    x: float
    y: float
    p: int
    d: int
    w: int
