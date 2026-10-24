"""Utilidades compartilhadas pelos testes."""

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from entities.instancia import carregar_instancia  # noqa: E402


def instancia(nome: str):
    """Carrega ``manutencao_<nome>.txt`` da raiz do projeto."""
    return carregar_instancia(os.path.join(RAIZ, f"manutencao_{nome}.txt"))
