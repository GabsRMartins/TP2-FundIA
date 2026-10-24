"""Algoritmo genetico geracional com elitismo.

Cromossomo: uma permutacao (a ordem de visita). Aptidao: ``h``, que e um
custo -- a selecao por torneio so compara aptidoes, entao basta declarar
vencedor o individuo de *menor* ``h``, sem converter o custo em outra medida.
"""

import random
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import numpy as np

from .metricas import OrcamentoEsgotado, ResultadoExecucao

if TYPE_CHECKING:  # pragma: no cover - apenas para anotacao de tipos
    from services.interface import ProblemaPermutacao


def diversidade_hamming(populacao: Sequence[Sequence[int]]) -> float:
    """Distancia de Hamming posicional media entre pares, normalizada por ``n``.

    Vale 0 quando todos os individuos sao iguais e fica perto de 1 numa
    populacao aleatoria. Para cada posicao, conta quantos pares tem o mesmo
    ativo ali (``sum c(c-1)/2`` sobre as contagens ``c`` de cada ativo), em
    ``O(N n)`` em vez de comparar os ``N(N-1)/2`` pares.
    """
    matriz = np.asarray(populacao, dtype=np.int64)
    tamanho, n = matriz.shape
    if tamanho < 2:
        return 0.0
    deslocamento = np.arange(n, dtype=np.int64) * (n + 1)
    contagens = np.bincount((matriz + deslocamento).ravel(), minlength=n * (n + 1))
    pares_iguais = float((contagens * (contagens - 1) // 2).sum())
    pares = tamanho * (tamanho - 1) / 2
    return 1.0 - pares_iguais / (pares * n)


def algoritmo_genetico(
    problema: "ProblemaPermutacao",
    rng: random.Random,
    tamanho_populacao: int = 100,
    pc: float = 0.9,
    pm: float = 0.2,
    k_torneio: int = 3,
    elite: int = 2,
    parceiro_aleatorio: bool = False,
) -> ResultadoExecucao:
    """Genetic algorithm com torneio, cruzamento, mutacao e elitismo.

    Ciclo de cada geracao: os ``elite`` melhores passam intactos; os demais
    filhos nascem de dois pais escolhidos por torneio de ``k_torneio``, com
    cruzamento com probabilidade ``pc`` e mutacao com probabilidade ``pm``.
    Um filho que nao sofreu nenhum dos dois e copia do primeiro pai e herda
    a aptidao dele, sem nova avaliacao. A parada e o fim do orcamento.

    ``parceiro_aleatorio`` implementa o controle da "galinha sem cabeca"
    (Jones, 1995): o segundo pai do cruzamento passa a ser uma permutacao
    aleatoria nova (nao avaliada), e nao um individuo da populacao. Se o GA
    nao piorar com isso, o cruzamento esta agindo como macromutacao, e nao
    combinando trechos bons de dois pais.
    """
    monitor = problema.monitor
    geracoes: Dict[str, List[float]] = {
        "avaliacoes": [], "melhor": [], "media": [], "distintos": [], "hamming": [],
    }
    contagem = {
        "filhos_cruzamento": 0, "filhos_cruzamento_melhores": 0,
        "filhos_mutacao": 0, "filhos_mutacao_melhores": 0,
    }

    def registrar(populacao, aptidoes) -> None:
        geracoes["avaliacoes"].append(monitor.avaliacoes)
        geracoes["melhor"].append(min(aptidoes))
        geracoes["media"].append(sum(aptidoes) / len(aptidoes))
        geracoes["distintos"].append(len({tuple(ind) for ind in populacao}))
        geracoes["hamming"].append(diversidade_hamming(populacao))

    def torneio(aptidoes) -> int:
        competidores = [rng.randrange(len(aptidoes)) for _ in range(k_torneio)]
        return min(competidores, key=aptidoes.__getitem__)

    populacao: List[List[int]] = []
    aptidoes: List[int] = []
    try:
        for _ in range(tamanho_populacao):
            individuo = problema.solucao_aleatoria()
            aptidoes.append(problema.h(individuo))
            populacao.append(individuo)
        registrar(populacao, aptidoes)

        while True:
            ordem = sorted(range(tamanho_populacao), key=aptidoes.__getitem__)
            nova = [populacao[i] for i in ordem[:elite]]
            novas_aptidoes = [aptidoes[i] for i in ordem[:elite]]
            while len(nova) < tamanho_populacao:
                i = torneio(aptidoes)
                j = torneio(aptidoes)
                filho = populacao[i]
                cruzou = rng.random() < pc
                mutou = rng.random() < pm
                if cruzou:
                    if parceiro_aleatorio:
                        parceiro, aptidao_parceiro = problema.solucao_aleatoria(), aptidoes[i]
                    else:
                        parceiro, aptidao_parceiro = populacao[j], aptidoes[j]
                    filho = problema.cruzamento(populacao[i], parceiro)
                if mutou:
                    filho = problema.mutacao(filho)
                if cruzou or mutou:
                    aptidao = problema.h(filho)
                    if cruzou and not mutou:
                        contagem["filhos_cruzamento"] += 1
                        if aptidao < min(aptidoes[i], aptidao_parceiro):
                            contagem["filhos_cruzamento_melhores"] += 1
                    elif mutou and not cruzou:
                        contagem["filhos_mutacao"] += 1
                        if aptidao < aptidoes[i]:
                            contagem["filhos_mutacao_melhores"] += 1
                else:
                    aptidao = aptidoes[i]
                nova.append(filho)
                novas_aptidoes.append(aptidao)
            populacao, aptidoes = nova, novas_aptidoes
            registrar(populacao, aptidoes)
    except OrcamentoEsgotado:
        pass

    extras: Dict[str, Any] = {
        "geracoes": len(geracoes["melhor"]) - 1,
        "distintos_final": geracoes["distintos"][-1] if geracoes["distintos"] else 0,
        "hamming_final": geracoes["hamming"][-1] if geracoes["hamming"] else 0.0,
        **contagem,
    }
    for tipo in ("cruzamento", "mutacao"):
        total = contagem[f"filhos_{tipo}"]
        extras[f"taxa_sucesso_{tipo}"] = (
            contagem[f"filhos_{tipo}_melhores"] / total if total else float("nan")
        )
    return ResultadoExecucao.do_monitor("GA", monitor, extras=extras, tracos=geracoes)
