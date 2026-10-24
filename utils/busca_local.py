"""Busca local: descida gulosa com random restart e tempera simulada.

Os algoritmos conhecem apenas a interface ``ProblemaPermutacao``. Cada um
roda ate o monitor do problema levantar ``OrcamentoEsgotado`` e devolve o
melhor valor visto, que o proprio monitor registra.
"""

import math
import random
import statistics
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .metricas import INFINITO, OrcamentoEsgotado, ResultadoExecucao

if TYPE_CHECKING:  # pragma: no cover - apenas para anotacao de tipos
    from services.interface import ProblemaPermutacao

VARIANTES = ("best", "first", "aleatorio")

#: Ate esse tamanho os otimos locais sao guardados para contar os distintos
#: entre execucoes (pergunta "Otimos locais" da Secao 5).
N_MAXIMO_GUARDAR_OTIMOS = 20


def descida_gulosa(
    problema: "ProblemaPermutacao",
    variante: str = "first",
    max_falhas: Optional[int] = None,
) -> ResultadoExecucao:
    """Greedy descent com random restart.

    ``variante``:

    * ``best``      -- varre toda a vizinhanca e move para o melhor sucessor;
    * ``first``     -- move para o primeiro sucessor que melhora ``h``;
    * ``aleatorio`` -- sorteia um sucessor e o aceita somente se melhorar.
      Como nao ha varredura, o otimo local e declarado apos ``max_falhas``
      sorteios seguidos sem melhora (padrao: o tamanho da vizinhanca).

    Ao atingir um otimo local, recomeca de uma permutacao aleatoria ate
    esgotar o orcamento.
    """
    if variante not in VARIANTES:
        raise ValueError(f"variante desconhecida: {variante!r} (use {VARIANTES})")
    if max_falhas is None:
        max_falhas = problema.tamanho_vizinhanca()

    passos = 0
    descidas = 0
    valores_otimos: List[int] = []
    otimos: set = set()
    passos_por_descida: List[int] = []

    try:
        while True:
            s = problema.solucao_aleatoria()
            hs = problema.h(s)
            descidas += 1
            passos_descida = 0
            falhas = 0
            while True:
                movimento = None
                if variante == "best":
                    melhor = hs
                    for mov in problema.sucessores(s):
                        valor = problema.avaliar_sucessor(s, mov, melhor)
                        if valor < melhor:
                            melhor, movimento = valor, mov
                elif variante == "first":
                    for mov in problema.sucessores(s):
                        valor = problema.avaliar_sucessor(s, mov, hs)
                        if valor < hs:
                            melhor, movimento = valor, mov
                            break
                else:
                    while falhas < max_falhas:
                        mov = problema.sucessor_aleatorio(s)
                        valor = problema.avaliar_sucessor(s, mov, hs)
                        if valor < hs:
                            melhor, movimento = valor, mov
                            falhas = 0
                            break
                        falhas += 1

                if movimento is None:  # otimo local
                    valores_otimos.append(hs)
                    passos_por_descida.append(passos_descida)
                    if len(s) <= N_MAXIMO_GUARDAR_OTIMOS:
                        otimos.add(tuple(s))
                    break
                s = problema.aplicar(s, movimento)
                hs = melhor
                passos += 1
                passos_descida += 1
    except OrcamentoEsgotado:
        pass

    extras: Dict[str, Any] = {
        "passos_melhoria": passos,
        "restarts": descidas,
        "otimos_locais": len(valores_otimos),
        "passos_por_descida": (
            statistics.fmean(passos_por_descida) if passos_por_descida else float("nan")
        ),
    }
    tracos: Dict[str, Any] = {"valores_otimos": valores_otimos}
    if otimos:
        extras["otimos_distintos"] = len(otimos)
        tracos["otimos"] = [" ".join(map(str, o)) for o in sorted(otimos)]
    return ResultadoExecucao.do_monitor(
        f"GD-{variante}", problema.monitor, extras=extras, tracos=tracos
    )


def calibrar_temperaturas(
    problema: "ProblemaPermutacao",
    amostras: int,
    aceitacao_inicial: float,
    aceitacao_final: float,
):
    """Estima ``T0`` e ``Tf`` a partir das pioras observadas num random walk.

    A escala certa de ``T`` e a escala de ``Delta = h(s') - h(s)``, que muda
    de uma instancia para outra. Um random walk de ``amostras`` passos (que
    conta no orcamento) mede essa escala na propria instancia:

    * ``T0 = -mediana(Delta+) / ln(aceitacao_inicial)``: a piora tipica e
      aceita com probabilidade ``aceitacao_inicial`` -- comeca como random walk;
    * ``Tf = -P10(Delta+) / ln(aceitacao_final)``: mesmo uma piora pequena
      (decil inferior) e quase sempre rejeitada -- termina como greedy descent.

    Devolve ``(T0, Tf, s, hs)``, com a solucao em que o passeio terminou.
    """
    s = problema.solucao_aleatoria()
    hs = problema.h(s)
    pioras: List[int] = []
    for _ in range(amostras):
        mov = problema.sucessor_aleatorio(s)
        valor = problema.avaliar_sucessor(s, mov)
        if valor > hs:
            pioras.append(valor - hs)
        s = problema.aplicar(s, mov)
        hs = valor
    if not pioras:
        pioras = [1]
    pioras.sort()
    tipica = statistics.median(pioras)
    pequena = pioras[int(0.1 * (len(pioras) - 1))]
    t0 = -tipica / math.log(aceitacao_inicial)
    tf = min(t0, -pequena / math.log(aceitacao_final))
    return t0, tf, s, hs


def tempera_simulada(
    problema: "ProblemaPermutacao",
    rng: random.Random,
    aceitacao_inicial: float = 0.95,
    aceitacao_final: float = 0.001,
    amostras_calibracao: int = 500,
    iteracoes_por_nivel: Optional[int] = None,
    janelas: int = 200,
) -> ResultadoExecucao:
    """Simulated annealing com resfriamento geometrico ``T <- alpha * T``.

    A cada iteracao sorteia um sucessor ``s'``: aceita se ``h(s') <= h(s)``;
    caso contrario, aceita com probabilidade ``exp(-(h(s') - h(s)) / T)``.

    Implementacao equivalente do teste de aceitacao: sorteia-se ``u`` antes
    da avaliacao e aceita-se ``s'`` se ``h(s') < h(s) - T ln u``. Como
    ``P(u < exp(-Delta/T)) = exp(-Delta/T)``, a regra e identica a de Gibbs,
    e o limiar conhecido antecipadamente permite interromper a avaliacao
    incremental de um sucessor que ja o ultrapassou.

    O schedule e calibrado por instancia (ver :func:`calibrar_temperaturas`)
    e ``alpha`` e escolhido para que ``T`` chegue a ``Tf`` exatamente quando
    o orcamento termina, com ``iteracoes_por_nivel`` (padrao: ``n``)
    iteracoes em cada temperatura.
    """
    monitor = problema.monitor
    pioras_aceitas = pioras_rejeitadas = melhoras = 0
    serie: Dict[str, List[float]] = {
        "avaliacoes": [], "temperatura": [], "taxa_aceitacao_pioras": [],
        "h_corrente": [], "melhor_h": [],
    }
    extras: Dict[str, Any] = {}

    try:
        t0, tf, s, hs = calibrar_temperaturas(
            problema, amostras_calibracao, aceitacao_inicial, aceitacao_final
        )
        nivel = iteracoes_por_nivel or len(s)
        niveis = max(1, int(monitor.restante) // nivel)
        alfa = (tf / t0) ** (1.0 / niveis)
        extras.update(T0=t0, Tf=tf, alfa=alfa, iteracoes_por_nivel=nivel)

        temperatura = t0
        iteracao = 0
        tamanho_janela = max(1, int(monitor.restante) // janelas)
        proxima_janela = monitor.avaliacoes + tamanho_janela
        aceitas_janela = pioras_janela = 0

        while True:
            mov = problema.sucessor_aleatorio(s)
            u = rng.random()
            limiar = hs - temperatura * math.log(u) if u > 0.0 else INFINITO
            valor = problema.avaliar_sucessor(s, mov, limiar)
            if valor < limiar:
                if valor > hs:
                    pioras_aceitas += 1
                    aceitas_janela += 1
                    pioras_janela += 1
                elif valor < hs:
                    melhoras += 1
                s = problema.aplicar(s, mov)
                hs = valor
            else:
                # Rejeicao implica piora: limiar > hs sempre que u < 1.
                pioras_rejeitadas += 1
                pioras_janela += 1

            iteracao += 1
            if iteracao % nivel == 0:
                temperatura *= alfa
            if monitor.avaliacoes >= proxima_janela:
                serie["avaliacoes"].append(monitor.avaliacoes)
                serie["temperatura"].append(temperatura)
                serie["taxa_aceitacao_pioras"].append(
                    aceitas_janela / pioras_janela if pioras_janela else float("nan")
                )
                serie["h_corrente"].append(hs)
                serie["melhor_h"].append(monitor.melhor_h)
                aceitas_janela = pioras_janela = 0
                proxima_janela += tamanho_janela
    except OrcamentoEsgotado:
        pass

    total_pioras = pioras_aceitas + pioras_rejeitadas
    extras.update(
        pioras_aceitas=pioras_aceitas,
        pioras_rejeitadas=pioras_rejeitadas,
        melhoras=melhoras,
        taxa_aceitacao_pioras=pioras_aceitas / total_pioras if total_pioras else float("nan"),
    )
    return ResultadoExecucao.do_monitor("SA", monitor, extras=extras, tracos=serie)
