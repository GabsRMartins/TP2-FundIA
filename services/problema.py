import random
from typing import Iterator, List, Optional, Sequence

from entities.instancia import Instancia
from entities.movimento import Movimento
from services.interface import ProblemaPermutacao
from utils import cruzamentos, vizinhancas
from utils.metricas import INFINITO, Monitor

CRUZAMENTOS = {"ox": cruzamentos.ox, "pmx": cruzamentos.pmx}


class ProblemaManutencao(ProblemaPermutacao):
    """Sequenciamento de manutencao de ativos com atraso ponderado total.

    ``vizinhanca``: operador que gera os sucessores (``swap``, ``insercao``
    ou ``2opt``). ``cruzamento`` (``ox`` ou ``pmx``) e ``mutacao`` (um dos
    operadores de vizinhanca) sao os operadores usados pelo algoritmo
    genetico.

    Avaliacao incremental: a ultima permutacao avaliada fica em cache junto
    com os instantes de conclusao ``C[k]`` e o custo acumulado ``H[k]`` de cada
    posicao ``k``. Um movimento sobre as posicoes ``i..j`` nao altera nada
    antes de ``min(i, j)``, entao o sucessor e avaliado a partir dali,
    retomando ``C`` e ``H`` do prefixo. Montar o cache nao conta como
    avaliacao: ele so e montado para uma permutacao cujo ``h`` ja foi contado
    quando ela foi gerada.
    """

    def __init__(
        self,
        instancia: Instancia,
        vizinhanca: str = vizinhancas.INSERCAO,
        rng: Optional[random.Random] = None,
        monitor: Optional[Monitor] = None,
        cruzamento: str = "ox",
        mutacao: str = vizinhancas.SWAP,
    ) -> None:
        if cruzamento not in CRUZAMENTOS:
            raise ValueError(f"cruzamento desconhecido: {cruzamento!r}")
        self.instancia = instancia
        self.n = instancia.n
        self.vizinhanca = vizinhanca
        self.rng = rng or random.Random()
        self.monitor = monitor or Monitor()
        self.operador_cruzamento = cruzamento
        self.operador_mutacao = mutacao

        self._movimentos = vizinhancas.enumerar(vizinhanca, self.n)
        self.rng.shuffle(self._movimentos)
        vizinhancas.tamanho(mutacao, self.n)  # valida o nome do operador
        self._cache_solucao: Optional[Sequence[int]] = None
        self._cache_c: List[int] = []
        self._cache_h: List[int] = []

    # ------------------------------------------------------------------
    # Geracao de solucoes
    # ------------------------------------------------------------------
    def solucao_aleatoria(self) -> List[int]:
        s = list(range(1, self.n + 1))
        self.rng.shuffle(s)
        return s

    def sucessores(self, s: Sequence[int]) -> Iterator[Movimento]:
        """Todos os movimentos, numa ordem aleatoria a partir de um ponto sorteado.

        A lista de movimentos e embaralhada uma vez, na construcao do
        problema, e cada varredura comeca num ponto sorteado dela. Na ordem
        lexicografica, movimentos consecutivos mexem no mesmo ativo (``i``
        fixo, ``j`` variando) e sao fortemente correlacionados: se mover o
        ativo ``i`` nao ajuda, o *first improvement* gastaria ``~n`` avaliacoes
        para descobrir isso. Embaralhar a cada passo custaria ``O(|N(s)|)``.
        """
        movimentos = self._movimentos
        inicio = self.rng.randrange(len(movimentos))
        yield from movimentos[inicio:]
        yield from movimentos[:inicio]

    def sucessor_aleatorio(self, s: Sequence[int]) -> Movimento:
        return vizinhancas.sortear(self.vizinhanca, self.n, self.rng)

    def tamanho_vizinhanca(self) -> int:
        return len(self._movimentos)

    def aplicar(self, s: Sequence[int], mov: Movimento) -> List[int]:
        return vizinhancas.aplicar(s, mov)

    def cruzamento(self, s1: Sequence[int], s2: Sequence[int]) -> List[int]:
        a, b = cruzamentos.sortear_cortes(self.n, self.rng)
        return CRUZAMENTOS[self.operador_cruzamento](s1, s2, a, b)

    def mutacao(self, s: Sequence[int]) -> List[int]:
        mov = vizinhancas.sortear(self.operador_mutacao, self.n, self.rng)
        return vizinhancas.aplicar(s, mov)

    # ------------------------------------------------------------------
    # Avaliacao
    # ------------------------------------------------------------------
    def _preparar(self, s: Sequence[int]) -> None:
        """Calcula ``C[k]`` e ``H[k]`` de cada posicao de ``s`` e guarda em cache."""
        inst = self.instancia
        t, p, d, w = inst.t, inst.p, inst.d, inst.w
        conclusoes = [0] * len(s)
        acumulado = [0] * len(s)
        conclusao = 0
        total = 0
        anterior = 0
        for k, j in enumerate(s):
            conclusao += t[anterior][j] + p[j]
            if conclusao > d[j]:
                total += w[j] * (conclusao - d[j])
            conclusoes[k] = conclusao
            acumulado[k] = total
            anterior = j
        self._cache_solucao = s
        self._cache_c = conclusoes
        self._cache_h = acumulado

    def h(self, s: Sequence[int]) -> int:
        self.monitor.consumir()
        self._preparar(s)
        valor = self._cache_h[-1] if self._cache_h else 0
        self.monitor.observar(valor, s)
        return valor

    def avaliar_sucessor(
        self, s: Sequence[int], mov: Movimento, limite: float = INFINITO
    ) -> int:
        self.monitor.consumir()
        if self._cache_solucao is not s:
            self._preparar(s)

        inicio = mov.i if mov.i < mov.j else mov.j
        fim = mov.j if mov.i < mov.j else mov.i
        if inicio > 0:
            conclusao = self._cache_c[inicio - 1]
            total = self._cache_h[inicio - 1]
            anterior = s[inicio - 1]
        else:
            conclusao = total = anterior = 0

        inst = self.instancia
        t, p, d, w = inst.t, inst.p, inst.d, inst.w
        for trecho in (vizinhancas.trecho(s, mov), s[fim + 1:]):
            for j in trecho:
                conclusao += t[anterior][j] + p[j]
                if conclusao > d[j]:
                    total += w[j] * (conclusao - d[j])
                    # h nunca diminui ao longo do prefixo: atingido o limite,
                    # o restante da sequencia nao pode trazer o valor de volta.
                    if total >= limite:
                        return total
                anterior = j

        self.monitor.observar(total, lambda: vizinhancas.aplicar(s, mov))
        return total
