"""Tabelas e testes estatisticos a partir dos CSVs gravados em ``resultados/``.

Tudo aqui le arquivos ja gravados: rodar a analise de novo nao executa
nenhum algoritmo.
"""

import os
import statistics
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

from services.executor import (
    DIRETORIO_RESULTADOS,
    INSTANCIAS,
    gravar_tabela,
    instancia,
    ler_csv,
    ler_tracos,
)
from utils.estatistica import a12, formatar_p, holm, mann_whitney, resumo

LARGURA = 100


def existe(nome: str) -> bool:
    return os.path.exists(os.path.join(DIRETORIO_RESULTADOS, f"{nome}.csv"))


def agrupar(linhas: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """``{instancia: {rotulo: [linhas]}}`` preservando a ordem de gravacao."""
    grupos: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for linha in linhas:
        grupos.setdefault(linha["instancia"], {}).setdefault(linha["rotulo"], []).append(linha)
    return grupos


def _numeros(linhas: Sequence[Dict[str, str]], coluna: str) -> List[float]:
    return [float(l[coluna]) for l in linhas if l.get(coluna, "") != ""]


def _media(linhas: Sequence[Dict[str, str]], coluna: str) -> Optional[float]:
    valores = _numeros(linhas, coluna)
    return statistics.fmean(valores) if valores else None


# ----------------------------------------------------------------------
# Tabelas genericas
# ----------------------------------------------------------------------
def tabela_resumo(
    linhas: Sequence[Dict[str, str]],
    extras: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Uma linha por ``instancia x rotulo``: estatisticas do melhor ``h`` e tempo."""
    tabela = []
    for nome, por_rotulo in agrupar(linhas).items():
        referencia = INSTANCIAS[nome].referencia
        for rotulo, grupo in por_rotulo.items():
            valores = _numeros(grupo, "melhor_h")
            estat = resumo(valores)
            linha: Dict[str, Any] = {
                "instancia": nome,
                "rotulo": rotulo,
                **{k: round(v, 2) for k, v in estat.items()},
                "gap_medio_%": round(100 * (estat["media"] - referencia) / referencia, 2),
                "gap_melhor_%": round(100 * (estat["minimo"] - referencia) / referencia, 2),
                "atingiu_referencia": sum(v <= referencia for v in valores),
                "avaliacoes": int(_media(grupo, "avaliacoes")),
                "tempo_medio_s": round(_media(grupo, "tempo_s"), 3),
                "tempo_dp_s": round(statistics.stdev(_numeros(grupo, "tempo_s")), 3)
                if len(grupo) > 1 else 0.0,
            }
            linha["us_por_avaliacao"] = round(1e6 * linha["tempo_medio_s"] / linha["avaliacoes"], 2)
            for coluna in extras:
                media = _media(grupo, coluna)
                linha[coluna] = round(media, 4) if media is not None else ""
            tabela.append(linha)
    return tabela


def testes_pareados(linhas: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Mann-Whitney entre todos os pares de rotulos de cada instancia.

    O p-valor e corrigido por Holm dentro de cada instancia. ``A12`` e a
    probabilidade de ``A`` encontrar ``h`` menor que ``B`` numa execucao.
    """
    testes = []
    for nome, por_rotulo in agrupar(linhas).items():
        pares = list(combinations(por_rotulo, 2))
        bloco = []
        for rotulo_a, rotulo_b in pares:
            a = _numeros(por_rotulo[rotulo_a], "melhor_h")
            b = _numeros(por_rotulo[rotulo_b], "melhor_h")
            u, p = mann_whitney(a, b)
            bloco.append({
                "instancia": nome, "A": rotulo_a, "B": rotulo_b,
                "mediana_A": statistics.median(a), "mediana_B": statistics.median(b),
                "U": u, "p": p, "A12": round(a12(a, b), 3),
            })
        for teste, ajustado in zip(bloco, holm([t["p"] for t in bloco])):
            teste["p_holm"] = ajustado
            if ajustado >= 0.05:
                teste["conclusao"] = "sem diferenca significativa"
            else:
                melhor = teste["A"] if teste["A12"] > 0.5 else teste["B"]
                teste["conclusao"] = f"{melhor} melhor"
        testes.extend(bloco)
    return testes


def imprimir(titulo: str, tabela: Sequence[Dict[str, Any]], colunas: Sequence[str]) -> None:
    print("\n" + titulo)
    print("-" * LARGURA)
    larguras = {
        c: max(len(c), *(len(_texto(l.get(c, ""))) for l in tabela)) + 2 for c in colunas
    }
    print("".join(f"{c:>{larguras[c]}}" if i else f"{c:<{larguras[c]}}"
                  for i, c in enumerate(colunas)))
    for linha in tabela:
        print("".join(
            f"{_texto(linha.get(c, '')):>{larguras[c]}}" if i
            else f"{_texto(linha.get(c, '')):<{larguras[c]}}"
            for i, c in enumerate(colunas)
        ))
    print("-" * LARGURA)


def _texto(valor: Any) -> str:
    if isinstance(valor, float):
        if valor != valor:  # nan
            return "--"
        if abs(valor) >= 1000:
            return f"{valor:,.0f}".replace(",", " ")
        return f"{valor:.4g}" if abs(valor) < 1 else f"{valor:.2f}"
    return str(valor)


def _com_p_formatado(testes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{**t, "p": formatar_p(t["p"]), "p_holm": formatar_p(t["p_holm"])} for t in testes]


COLUNAS_TESTE = ["instancia", "A", "B", "mediana_A", "mediana_B", "p", "p_holm", "A12", "conclusao"]


# ----------------------------------------------------------------------
# Analises de cada experimento
# ----------------------------------------------------------------------
def analisar_vizinhancas() -> None:
    linhas = ler_csv("vizinhancas")
    tabela = tabela_resumo(linhas, extras=("passos_melhoria", "otimos_locais"))
    imprimir("T2 -- GREEDY DESCENT (FIRST IMPROVEMENT) POR VIZINHANCA", tabela,
             ["instancia", "rotulo", "media", "dp", "mediana", "minimo", "gap_medio_%",
              "passos_melhoria", "otimos_locais", "us_por_avaliacao"])
    testes = testes_pareados(linhas)
    imprimir("T2 -- TESTES DE MANN-WHITNEY ENTRE VIZINHANCAS", _com_p_formatado(testes),
             COLUNAS_TESTE)
    gravar_tabela("tabela_vizinhancas", tabela)
    gravar_tabela("testes_vizinhancas", testes)

    if existe("avaliacao_incremental"):
        tempos = ler_csv("avaliacao_incremental")
        imprimir("CUSTO POR AVALIACAO (microssegundos)", tempos,
                 ["instancia", "vizinhanca", "contexto", "us_completa", "us_incremental",
                  "us_incremental_corte", "ganho_incremental", "ganho_incremental_corte"])


def analisar_variantes() -> None:
    linhas = ler_csv("variantes")
    tabela = tabela_resumo(
        linhas,
        extras=("passos_melhoria", "restarts", "otimos_locais", "passos_por_descida",
                "otimos_distintos"),
    )
    imprimir("VARIANTES DO GREEDY DESCENT (passos de melhoria dentro do orcamento)", tabela,
             ["instancia", "rotulo", "media", "dp", "mediana", "minimo", "passos_melhoria",
              "restarts", "otimos_locais", "passos_por_descida", "otimos_distintos"])
    testes = testes_pareados(linhas)
    imprimir("VARIANTES -- TESTES DE MANN-WHITNEY", _com_p_formatado(testes), COLUNAS_TESTE)
    gravar_tabela("tabela_variantes", tabela)
    gravar_tabela("testes_variantes", testes)


def analisar_calibracao_ga() -> None:
    linhas = ler_csv("calibracao_ga")
    tabela = tabela_resumo(linhas, extras=("hamming_final", "distintos_final", "geracoes"))
    tabela.sort(key=lambda l: l["mediana"])
    imprimir("CALIBRACAO DO GA (instancia media, ordenado pela mediana)", tabela,
             ["rotulo", "media", "dp", "mediana", "minimo", "hamming_final",
              "distintos_final", "geracoes"])
    gravar_tabela("tabela_calibracao_ga", tabela)


def analisar_principal() -> None:
    linhas = ler_csv("experimentos")
    tabela = tabela_resumo(linhas)
    imprimir("TABELA PRINCIPAL (melhor h por execucao)", tabela,
             ["instancia", "rotulo", "media", "dp", "minimo", "mediana", "maximo",
              "gap_medio_%", "gap_melhor_%", "atingiu_referencia"])
    imprimir("TEMPO x AVALIACOES", tabela,
             ["instancia", "rotulo", "avaliacoes", "tempo_medio_s", "tempo_dp_s",
              "us_por_avaliacao"])
    testes = testes_pareados(linhas)
    imprimir("TESTES DE MANN-WHITNEY (bilateral, Holm por instancia)",
             _com_p_formatado(testes), COLUNAS_TESTE)
    gravar_tabela("tabela_principal", tabela)
    gravar_tabela("testes_principais", testes)
    analisar_otimos_locais(linhas)


def analisar_otimos_locais(linhas_principais: Sequence[Dict[str, str]]) -> None:
    """Pergunta "Otimos locais" da Secao 5, na instancia pequena."""
    referencia = INSTANCIAS["pequena"].referencia
    tabela = []
    fontes = [("experimentos", ler_tracos("experimentos"))]
    if existe("variantes"):
        fontes.append(("variantes", ler_tracos("variantes")))
    for experimento, tracos in fontes:
        for chave, execucoes in tracos.items():
            nome, rotulo = chave.split("|")
            if nome != "pequena" or "otimos" not in execucoes[0]:
                continue
            distintos = set()
            valores = []
            for execucao in execucoes:
                distintos.update(execucao["otimos"])
                valores.extend(execucao["valores_otimos"])
            tabela.append({
                "experimento": experimento, "rotulo": rotulo,
                "descidas_completas": len(valores),
                "otimos_distintos": len(distintos),
                "otimos_distintos_globais": sum(
                    1 for o in distintos if _h_pequena(o) == referencia),
                "descidas_no_otimo_%": round(
                    100 * sum(v == referencia for v in valores) / len(valores), 2),
                "pior_otimo_local": max(valores),
            })
    acertos = []
    for nome_rotulo, grupo in agrupar(linhas_principais).get("pequena", {}).items():
        valores = _numeros(grupo, "melhor_h")
        acertos.append({"rotulo": nome_rotulo, "execucoes": len(valores),
                        "execucoes_no_otimo": sum(v == referencia for v in valores)})
    imprimir("OTIMOS LOCAIS NA INSTANCIA PEQUENA (todas as execucoes somadas)", tabela,
             ["experimento", "rotulo", "descidas_completas", "otimos_distintos",
              "otimos_distintos_globais", "descidas_no_otimo_%", "pior_otimo_local"])
    imprimir("EXECUCOES QUE ATINGIRAM h* = 1875", acertos,
             ["rotulo", "execucoes", "execucoes_no_otimo"])
    gravar_tabela("otimos_locais_pequena", tabela)


def _h_pequena(texto: str) -> int:
    return instancia("pequena").atraso_ponderado([int(x) for x in texto.split()])


def analisar_hipotese_ga() -> None:
    linhas = ler_csv("ga_hipotese")
    tabela = tabela_resumo(
        linhas,
        extras=("taxa_sucesso_cruzamento", "taxa_sucesso_mutacao", "hamming_final",
                "distintos_final"),
    )
    imprimir("HIPOTESE: A RECOMBINACAO COMBINA BONS TRECHOS?", tabela,
             ["instancia", "rotulo", "media", "dp", "mediana", "minimo",
              "taxa_sucesso_cruzamento", "taxa_sucesso_mutacao", "hamming_final"])
    testes = testes_pareados(linhas)
    imprimir("HIPOTESE -- TESTES DE MANN-WHITNEY", _com_p_formatado(testes), COLUNAS_TESTE)
    gravar_tabela("tabela_hipotese_ga", tabela)
    gravar_tabela("testes_hipotese_ga", testes)


#: A mesma configuracao do GD (first improvement, insercao) aparece em tres
#: experimentos, cada um com suas proprias sementes.
REPLICAS_GD = (("vizinhancas", "GD-first/insercao"), ("variantes", "GD-first"),
               ("experimentos", "GD"))


def analisar_replicas_gd() -> None:
    """Variacao amostral: tres amostras independentes da mesma configuracao."""
    amostras = {}
    for experimento, rotulo in REPLICAS_GD:
        for nome, por_rotulo in agrupar(ler_csv(experimento)).items():
            if rotulo in por_rotulo:
                amostras.setdefault(nome, {})[experimento] = _numeros(por_rotulo[rotulo], "melhor_h")
    tabela = []
    for nome, por_experimento in amostras.items():
        for a, b in combinations(por_experimento, 2):
            u, p = mann_whitney(por_experimento[a], por_experimento[b])
            tabela.append({
                "instancia": nome, "A": a, "B": b,
                "mediana_A": statistics.median(por_experimento[a]),
                "mediana_B": statistics.median(por_experimento[b]),
                "p": p, "A12": round(a12(por_experimento[a], por_experimento[b]), 3),
            })
    imprimir("REPLICAS DO GD FIRST/INSERCAO EM EXPERIMENTOS DIFERENTES (mesma configuracao)",
             [{**t, "p": formatar_p(t["p"])} for t in tabela],
             ["instancia", "A", "B", "mediana_A", "mediana_B", "p", "A12"])
    gravar_tabela("replicas_gd", tabela)


ANALISES = {
    "vizinhancas": analisar_vizinhancas,
    "variantes": analisar_variantes,
    "calibracao_ga": analisar_calibracao_ga,
    "experimentos": analisar_principal,
    "ga_hipotese": analisar_hipotese_ga,
}


def analisar_tudo() -> None:
    for nome, funcao in ANALISES.items():
        if existe(nome):
            funcao()
        else:
            print(f"\n(sem resultados/{nome}.csv -- rode `python main.py {nome.replace('_', '-')}`)")
    if all(existe(experimento) for experimento, _ in REPLICAS_GD):
        analisar_replicas_gd()
