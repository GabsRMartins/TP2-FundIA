"""Execucao dos experimentos e gravacao dos resultados em disco.

Cada experimento e uma lista de configuracoes (algoritmo + parametros)
rodada sobre as instancias com o orcamento e o numero de execucoes da Secao 3
do enunciado. As execucoes sao independentes e rodam em paralelo; cada uma
tem semente propria, derivada de ``(experimento, instancia, rotulo,
execucao)``, de modo que o mesmo comando reproduz os mesmos numeros.

Saidas em ``resultados/``:

* ``<experimento>.csv``        -- uma linha por execucao (melhor h, tempo...);
* ``<experimento>_tracos.json`` -- curva de convergencia e series de cada execucao.
"""

import csv
import functools
import json
import math
import os
import random
import time
import zlib
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Optional, Sequence

from entities.instancia import Instancia, carregar_instancia
from services.problema import ProblemaManutencao
from utils import vizinhancas
from utils.busca_local import descida_gulosa, tempera_simulada
from utils.genetico import algoritmo_genetico
from utils.metricas import Monitor, ResultadoExecucao

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRETORIO_RESULTADOS = os.path.join(RAIZ, "resultados")


@dataclass(frozen=True)
class Protocolo:
    """Instancia com orcamento, execucoes e valor de referencia (Tabela 1)."""

    arquivo: str
    orcamento: int
    execucoes: int
    referencia: int


INSTANCIAS: Dict[str, Protocolo] = {
    "pequena": Protocolo("manutencao_pequena.txt", 50_000, 30, 1875),
    "media": Protocolo("manutencao_media.txt", 150_000, 30, 16_332),
    "grande": Protocolo("manutencao_grande.txt", 300_000, 10, 60_417),
}

#: Vizinhanca adotada apos a comparacao da T2 (ver README).
VIZINHANCA_PADRAO = vizinhancas.INSERCAO
#: Variante do greedy descent adotada apos o experimento ``variantes``.
VARIANTE_PADRAO = "first"
#: Parametros do GA adotados apos o experimento ``calibracao-ga``.
PARAMETROS_GA: Dict[str, Any] = {
    "tamanho_populacao": 100, "k_torneio": 3, "elite": 2, "pc": 0.6, "pm": 0.5,
}
MUTACAO_GA = vizinhancas.INSERCAO

#: Pontos maximos guardados por serie de geracoes do GA.
MAX_PONTOS_SERIE = 600


@dataclass
class Configuracao:
    """Um algoritmo com seus parametros, identificado por ``rotulo``."""

    rotulo: str
    algoritmo: str  # "gd", "sa" ou "ga"
    vizinhanca: str = VIZINHANCA_PADRAO
    parametros: Dict[str, Any] = field(default_factory=dict)
    cruzamento: str = "ox"
    mutacao: str = MUTACAO_GA


def gd(variante: str = VARIANTE_PADRAO, vizinhanca: str = VIZINHANCA_PADRAO,
       rotulo: str = "GD") -> Configuracao:
    return Configuracao(rotulo, "gd", vizinhanca, {"variante": variante})


def sa(rotulo: str = "SA", **parametros: Any) -> Configuracao:
    return Configuracao(rotulo, "sa", parametros=parametros)


def ga(rotulo: str = "GA", cruzamento: str = "ox", mutacao: str = MUTACAO_GA,
       **parametros: Any) -> Configuracao:
    return Configuracao(rotulo, "ga", parametros={**PARAMETROS_GA, **parametros},
                        cruzamento=cruzamento, mutacao=mutacao)


# ----------------------------------------------------------------------
# Uma execucao
# ----------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def instancia(nome: str) -> Instancia:
    return carregar_instancia(os.path.join(RAIZ, INSTANCIAS[nome].arquivo))


def semente(*partes: Any) -> int:
    """Semente estavel entre execucoes do Python (``hash`` de str nao e)."""
    return zlib.crc32("|".join(map(str, partes)).encode("utf-8"))


def executar_uma(
    configuracao: Configuracao,
    nome_instancia: str,
    orcamento: int,
    semente_execucao: int,
) -> ResultadoExecucao:
    """Roda uma execucao independente e mede o tempo de parede."""
    rng = random.Random(semente_execucao)
    problema = ProblemaManutencao(
        instancia(nome_instancia),
        vizinhanca=configuracao.vizinhanca,
        rng=rng,
        monitor=Monitor(orcamento),
        cruzamento=configuracao.cruzamento,
        mutacao=configuracao.mutacao,
    )
    inicio = time.perf_counter()
    if configuracao.algoritmo == "gd":
        resultado = descida_gulosa(problema, **configuracao.parametros)
    elif configuracao.algoritmo == "sa":
        resultado = tempera_simulada(problema, rng, **configuracao.parametros)
    elif configuracao.algoritmo == "ga":
        resultado = algoritmo_genetico(problema, rng, **configuracao.parametros)
    else:
        raise ValueError(f"algoritmo desconhecido: {configuracao.algoritmo!r}")
    resultado.tempo = time.perf_counter() - inicio
    return resultado


def _tarefa(argumentos) -> Dict[str, Any]:
    experimento, configuracao, nome_instancia, orcamento, execucao = argumentos
    s = semente(experimento, nome_instancia, configuracao.rotulo, execucao)
    resultado = executar_uma(configuracao, nome_instancia, orcamento, s)
    linha = {
        "experimento": experimento,
        "instancia": nome_instancia,
        "n": instancia(nome_instancia).n,
        "rotulo": configuracao.rotulo,
        "algoritmo": resultado.algoritmo,
        "vizinhanca": configuracao.vizinhanca,
        "execucao": execucao,
        "semente": s,
        "orcamento": orcamento,
        "avaliacoes": resultado.avaliacoes,
        "melhor_h": resultado.melhor_h,
        "tempo_s": round(resultado.tempo, 4),
        **{chave: _arredondar(valor) for chave, valor in resultado.extras.items()},
        "melhor_solucao": " ".join(map(str, resultado.melhor_solucao)),
    }
    return {
        "linha": linha,
        "curva": resultado.curva,
        "tracos": _reduzir_series(resultado.tracos),
    }


def _arredondar(valor: Any) -> Any:
    if isinstance(valor, float):
        return round(valor, 6) if math.isfinite(valor) else ""
    return valor


def _reduzir_series(tracos: Dict[str, Any]) -> Dict[str, Any]:
    """Subamostra as series indexadas por ``avaliacoes`` (geracoes do GA)."""
    eixo = tracos.get("avaliacoes")
    if not isinstance(eixo, list) or len(eixo) <= MAX_PONTOS_SERIE:
        return tracos
    comprimento = len(eixo)
    passo = math.ceil(comprimento / MAX_PONTOS_SERIE)
    indices = list(range(0, comprimento, passo))
    if indices[-1] != comprimento - 1:
        indices.append(comprimento - 1)
    return {
        chave: [valor[i] for i in indices] if isinstance(valor, list) and len(valor) == comprimento
        else valor
        for chave, valor in tracos.items()
    }


# ----------------------------------------------------------------------
# Um experimento
# ----------------------------------------------------------------------
def rodar_experimento(
    experimento: str,
    configuracoes: Sequence[Configuracao],
    instancias: Iterable[str] = tuple(INSTANCIAS),
    rapido: bool = False,
    processos: Optional[int] = None,
    execucoes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Roda todas as combinacoes ``configuracao x instancia x execucao``.

    ``rapido`` divide o orcamento por 10 e usa 3 execucoes (teste de fumaca);
    os arquivos gravados recebem o sufixo ``_rapido`` para nao sobrescrever
    os resultados completos.
    """
    tarefas = []
    for nome in instancias:
        protocolo = INSTANCIAS[nome]
        orcamento = protocolo.orcamento // 10 if rapido else protocolo.orcamento
        total = 3 if rapido else (execucoes or protocolo.execucoes)
        for configuracao in configuracoes:
            for execucao in range(total):
                tarefas.append((experimento, configuracao, nome, orcamento, execucao))

    # As tarefas mais caras (instancia grande) primeiro equilibram melhor o pool.
    tarefas.sort(key=lambda t: -t[3])
    processos = processos or max(1, (os.cpu_count() or 2) - 1)
    inicio = time.perf_counter()
    print(f"[{experimento}] {len(tarefas)} execucoes em {processos} processos...", flush=True)
    if processos == 1:
        saidas = [_tarefa(t) for t in tarefas]
    else:
        with Pool(processos) as pool:
            saidas = pool.map(_tarefa, tarefas, chunksize=1)
    print(f"[{experimento}] concluido em {time.perf_counter() - inicio:.1f} s", flush=True)

    ordem = {nome: k for k, nome in enumerate(INSTANCIAS)}
    rotulos = {c.rotulo: k for k, c in enumerate(configuracoes)}
    saidas.sort(key=lambda s: (ordem[s["linha"]["instancia"]], rotulos[s["linha"]["rotulo"]],
                               s["linha"]["execucao"]))
    gravar(experimento + ("_rapido" if rapido else ""), saidas)
    return [s["linha"] for s in saidas]


def gravar(nome: str, saidas: List[Dict[str, Any]]) -> None:
    os.makedirs(DIRETORIO_RESULTADOS, exist_ok=True)
    linhas = [s["linha"] for s in saidas]
    colunas: List[str] = []
    for linha in linhas:
        colunas.extend(c for c in linha if c not in colunas)
    colunas.remove("melhor_solucao")
    colunas.append("melhor_solucao")

    caminho_csv = os.path.join(DIRETORIO_RESULTADOS, f"{nome}.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas, restval="")
        escritor.writeheader()
        escritor.writerows(linhas)

    tracos: Dict[str, List[Dict[str, Any]]] = {}
    for saida in saidas:
        linha = saida["linha"]
        tracos.setdefault(f"{linha['instancia']}|{linha['rotulo']}", []).append(
            {"execucao": linha["execucao"], "curva": saida["curva"], **saida["tracos"]}
        )
    caminho_json = os.path.join(DIRETORIO_RESULTADOS, f"{nome}_tracos.json")
    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(tracos, arquivo, separators=(",", ":"))
    print(f"gravado: {caminho_csv}\ngravado: {caminho_json}", flush=True)


def ler_csv(nome: str) -> List[Dict[str, str]]:
    with open(os.path.join(DIRETORIO_RESULTADOS, f"{nome}.csv"), encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def ler_tracos(nome: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(os.path.join(DIRETORIO_RESULTADOS, f"{nome}_tracos.json"), encoding="utf-8") as arq:
        return json.load(arq)


# ----------------------------------------------------------------------
# Configuracoes de cada experimento
# ----------------------------------------------------------------------
def configuracoes_vizinhancas() -> List[Configuracao]:
    """T2: greedy descent (first improvement) com cada vizinhanca."""
    return [gd("first", tipo, rotulo=f"GD-first/{tipo}") for tipo in vizinhancas.VIZINHANCAS]


def configuracoes_variantes() -> List[Configuracao]:
    """Secao 2.3: as tres variantes de selecao do proximo sucessor."""
    return [gd(variante, rotulo=f"GD-{variante}") for variante in ("best", "first", "aleatorio")]


def configuracoes_calibracao_ga() -> List[Configuracao]:
    """Grade de parametros do GA (rodada so na instancia media)."""
    return [
        ga(f"GA/{mutacao}/pc={pc}/pm={pm}", mutacao=mutacao, pc=pc, pm=pm)
        for mutacao in (vizinhancas.SWAP, vizinhancas.INSERCAO)
        for pc in (0.3, 0.6, 0.9)
        for pm in (0.2, 0.5, 1.0)
    ]


def configuracoes_principais() -> List[Configuracao]:
    """T3: os tres algoritmos com a vizinhanca e os parametros adotados."""
    return [gd(), sa(), ga()]


def configuracoes_hipotese_ga() -> List[Configuracao]:
    """Secao 5: a recombinacao combina bons trechos dos pais?

    * ``GA-OX``: o GA principal;
    * ``GA-PMX``: troca o OX por um cruzamento que preserva posicoes absolutas;
    * ``GA-OX-aleatorio``: teste da "galinha sem cabeca" (Jones, 1995) -- o
      segundo pai e uma permutacao aleatoria nova, entao o cruzamento vira
      uma macromutacao sem informacao da populacao;
    * ``GA-sem-cruzamento``: ``pc = 0`` e toda crianca e um pai mutado.
    """
    return [
        ga("GA-OX"),
        ga("GA-PMX", cruzamento="pmx"),
        ga("GA-OX-aleatorio", parceiro_aleatorio=True),
        ga("GA-sem-cruzamento", pc=0.0, pm=1.0),
    ]


# ----------------------------------------------------------------------
# Custo da avaliacao incremental (T2 e Secao 3)
# ----------------------------------------------------------------------
def medir_avaliacao(amostras: int = 20_000, semente_base: int = 2026) -> List[Dict[str, Any]]:
    """Tempo medio por avaliacao: completa x incremental x incremental com corte.

    Medido em dois contextos: numa permutacao aleatoria e num otimo local
    (obtido por first improvement), onde passa a maior parte das avaliacoes
    de uma busca local. O limite do corte e o ``h`` da solucao corrente, como
    na descida gulosa. Roda num unico processo para nao disputar CPU.
    """
    linhas = []
    for nome in INSTANCIAS:
        inst = instancia(nome)
        for tipo in vizinhancas.VIZINHANCAS:
            rng = random.Random(semente(semente_base, nome, tipo))
            problema = ProblemaManutencao(inst, vizinhanca=tipo, rng=rng)
            s = problema.solucao_aleatoria()
            for contexto in ("aleatoria", "otimo_local"):
                if contexto == "otimo_local":
                    s = descida_completa(problema, s)
                hs = inst.atraso_ponderado(s)
                movimentos = [problema.sucessor_aleatorio(s) for _ in range(amostras)]
                tempos = {}
                for modo in ("completa", "incremental", "incremental_corte"):
                    problema.h(s)  # deixa o cache pronto para s
                    inicio = time.perf_counter()
                    if modo == "completa":
                        for mov in movimentos:
                            inst.atraso_ponderado(vizinhancas.aplicar(s, mov))
                    elif modo == "incremental":
                        for mov in movimentos:
                            problema.avaliar_sucessor(s, mov)
                    else:
                        for mov in movimentos:
                            problema.avaliar_sucessor(s, mov, hs)
                    tempos[modo] = (time.perf_counter() - inicio) / amostras * 1e6
                linhas.append({
                    "instancia": nome, "n": inst.n, "vizinhanca": tipo, "contexto": contexto,
                    "h_solucao": hs,
                    "us_completa": round(tempos["completa"], 3),
                    "us_incremental": round(tempos["incremental"], 3),
                    "us_incremental_corte": round(tempos["incremental_corte"], 3),
                    "ganho_incremental": round(tempos["completa"] / tempos["incremental"], 3),
                    "ganho_incremental_corte": round(
                        tempos["completa"] / tempos["incremental_corte"], 3),
                })
                print(f"  {nome:8}{tipo:9}{contexto:12} completa {tempos['completa']:6.2f} us | "
                      f"incremental {tempos['incremental']:6.2f} us | "
                      f"com corte {tempos['incremental_corte']:6.2f} us", flush=True)
    return linhas


def descida_completa(problema: ProblemaManutencao, s: List[int]) -> List[int]:
    """First improvement ate um otimo local, sem limite de orcamento."""
    hs = problema.h(s)
    melhorou = True
    while melhorou:
        melhorou = False
        for mov in problema.sucessores(s):
            valor = problema.avaliar_sucessor(s, mov, hs)
            if valor < hs:
                s, hs, melhorou = problema.aplicar(s, mov), valor, True
                break
    return s


def gravar_tabela(nome: str, linhas: List[Dict[str, Any]]) -> str:
    """Grava uma tabela simples (lista de dicionarios) em ``resultados/<nome>.csv``."""
    os.makedirs(DIRETORIO_RESULTADOS, exist_ok=True)
    caminho = os.path.join(DIRETORIO_RESULTADOS, f"{nome}.csv")
    colunas: List[str] = []
    for linha in linhas:
        colunas.extend(c for c in linha if c not in colunas)
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas, restval="")
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"gravado: {caminho}", flush=True)
    return caminho
