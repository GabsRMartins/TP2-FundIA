"""A avaliacao incremental concorda com a avaliacao completa."""

import random
import unittest

from tests.comum import instancia

from services.problema import ProblemaManutencao
from utils import vizinhancas
from utils.metricas import Monitor

AMOSTRAS = 3000


class TestAvaliacaoIncremental(unittest.TestCase):
    def setUp(self):
        self.instancias = [instancia(nome) for nome in ("pequena", "media", "grande")]

    def _problema(self, inst, tipo, semente=0):
        return ProblemaManutencao(inst, vizinhanca=tipo, rng=random.Random(semente))

    def test_h_do_problema_igual_a_referencia(self):
        rng = random.Random(1)
        for inst in self.instancias:
            problema = self._problema(inst, vizinhancas.SWAP)
            for _ in range(200):
                s = list(range(1, inst.n + 1))
                rng.shuffle(s)
                self.assertEqual(problema.h(s), inst.atraso_ponderado(s))

    def test_incremental_sem_limite_e_exato(self):
        for inst in self.instancias:
            for tipo in vizinhancas.VIZINHANCAS:
                with self.subTest(instancia=inst.nome, vizinhanca=tipo):
                    problema = self._problema(inst, tipo)
                    s = problema.solucao_aleatoria()
                    for k in range(AMOSTRAS):
                        if k % 100 == 0:  # troca a solucao de referencia (e o cache)
                            s = problema.solucao_aleatoria()
                        mov = problema.sucessor_aleatorio(s)
                        esperado = inst.atraso_ponderado(problema.aplicar(s, mov))
                        self.assertEqual(problema.avaliar_sucessor(s, mov), esperado)

    def test_incremental_com_limite(self):
        """Abaixo do limite o valor e exato; acima, o retorno ja atinge o limite."""
        for inst in self.instancias:
            for tipo in vizinhancas.VIZINHANCAS:
                with self.subTest(instancia=inst.nome, vizinhanca=tipo):
                    problema = self._problema(inst, tipo)
                    s = problema.solucao_aleatoria()
                    limite = problema.h(s)
                    for _ in range(AMOSTRAS):
                        mov = problema.sucessor_aleatorio(s)
                        exato = inst.atraso_ponderado(problema.aplicar(s, mov))
                        valor = problema.avaliar_sucessor(s, mov, limite)
                        if exato < limite:
                            self.assertEqual(valor, exato)
                        else:
                            self.assertGreaterEqual(valor, limite)
                            self.assertLessEqual(valor, exato)

    def test_cada_avaliacao_conta_uma_vez(self):
        inst = self.instancias[0]
        monitor = Monitor()
        problema = ProblemaManutencao(inst, rng=random.Random(0), monitor=monitor)
        s = problema.solucao_aleatoria()
        problema.h(s)
        for mov in list(problema.sucessores(s)):
            problema.avaliar_sucessor(s, mov, limite=0)
        self.assertEqual(monitor.avaliacoes, 1 + problema.tamanho_vizinhanca())

    def test_melhor_registrado_e_o_menor_valor_exato(self):
        inst = self.instancias[1]
        monitor = Monitor()
        problema = ProblemaManutencao(inst, rng=random.Random(3), monitor=monitor)
        s = problema.solucao_aleatoria()
        valores = [problema.h(s)]
        for mov in problema.sucessores(s):
            valores.append(problema.avaliar_sucessor(s, mov))
        self.assertEqual(monitor.melhor_h, min(valores))
        self.assertEqual(inst.atraso_ponderado(monitor.melhor_solucao), min(valores))


if __name__ == "__main__":
    unittest.main()
