"""Estatisticas descritivas e testes nao parametricos (Secao 3 do enunciado)."""

import math
import statistics
from typing import Dict, List, Sequence, Tuple

from scipy.stats import mannwhitneyu


def resumo(valores: Sequence[float]) -> Dict[str, float]:
    """Media, desvio-padrao amostral, minimo, mediana e maximo."""
    valores = list(valores)
    return {
        "execucoes": len(valores),
        "media": statistics.fmean(valores),
        "dp": statistics.stdev(valores) if len(valores) > 1 else 0.0,
        "minimo": min(valores),
        "mediana": statistics.median(valores),
        "maximo": max(valores),
    }


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Teste de Mann-Whitney bilateral: ``(U de a, p-valor)``.

    Se as duas amostras forem inteiramente iguais a uma mesma constante (por
    exemplo, todas as execucoes no otimo), nao ha evidencia de diferenca e o
    p-valor e 1.
    """
    if len(set(a) | set(b)) == 1:
        return len(a) * len(b) / 2, 1.0
    teste = mannwhitneyu(a, b, alternative="two-sided")
    return float(teste.statistic), float(teste.pvalue)


def a12(a: Sequence[float], b: Sequence[float]) -> float:
    """Medida de efeito de Vargha-Delaney para minimizacao.

    Probabilidade de uma execucao de ``a`` encontrar ``h`` menor que uma
    execucao de ``b`` (empates contam meio). 0.5 = sem diferenca; acima de
    0.71 ou abaixo de 0.29 costuma ser lido como efeito grande.
    """
    vitorias = sum((x < y) + 0.5 * (x == y) for x in a for y in b)
    return vitorias / (len(a) * len(b))


def holm(pvalores: Sequence[float]) -> List[float]:
    """Correcao de Holm-Bonferroni para comparacoes multiplas."""
    m = len(pvalores)
    ordem = sorted(range(m), key=lambda k: pvalores[k])
    ajustados = [0.0] * m
    maximo = 0.0
    for posicao, k in enumerate(ordem):
        maximo = max(maximo, min(1.0, (m - posicao) * pvalores[k]))
        ajustados[k] = maximo
    return ajustados


def formatar_p(p: float) -> str:
    if math.isnan(p):
        return "--"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
