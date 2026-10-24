"""Operadores de vizinhanca, de cruzamento e o controle do orcamento."""

import random
import unittest
from collections import Counter

from tests.comum import instancia

from entities.movimento import Movimento
from services.problema import ProblemaManutencao
from utils import cruzamentos, vizinhancas
from utils.metricas import Monitor, OrcamentoEsgotado

EXEMPLO = [3, 7, 1, 5, 2, 8]


class TestVizinhancas(unittest.TestCase):
    def test_exemplos_da_figura_3(self):
        aplicar = vizinhancas.aplicar
        self.assertEqual(aplicar(EXEMPLO, Movimento("swap", 1, 4)), [3, 2, 1, 5, 7, 8])
        self.assertEqual(aplicar(EXEMPLO, Movimento("insercao", 1, 4)), [3, 1, 5, 2, 7, 8])
        self.assertEqual(aplicar(EXEMPLO, Movimento("2opt", 1, 4)), [3, 2, 5, 1, 7, 8])

    def test_insercao_para_tras(self):
        self.assertEqual(
            vizinhancas.aplicar(EXEMPLO, Movimento("insercao", 4, 1)), [3, 2, 7, 1, 5, 8]
        )

    def test_tamanho_da_vizinhanca_conta_sucessores_distintos(self):
        for n in (2, 3, 6, 12):
            s = list(range(1, n + 1))
            for tipo in vizinhancas.VIZINHANCAS:
                with self.subTest(n=n, vizinhanca=tipo):
                    movimentos = vizinhancas.enumerar(tipo, n)
                    sucessores = {tuple(vizinhancas.aplicar(s, m)) for m in movimentos}
                    self.assertEqual(len(movimentos), vizinhancas.tamanho(tipo, n))
                    self.assertEqual(len(sucessores), len(movimentos))
                    self.assertNotIn(tuple(s), sucessores)

    def test_tamanhos_na_instancia_grande(self):
        self.assertEqual(vizinhancas.tamanho("swap", 150), 11175)
        self.assertEqual(vizinhancas.tamanho("2opt", 150), 11175)
        self.assertEqual(vizinhancas.tamanho("insercao", 150), 149 ** 2)

    def test_sorteio_uniforme_sobre_movimentos_distintos(self):
        rng = random.Random(7)
        n = 5
        for tipo in vizinhancas.VIZINHANCAS:
            with self.subTest(vizinhanca=tipo):
                validos = set(vizinhancas.enumerar(tipo, n))
                contagem = Counter(vizinhancas.sortear(tipo, n, rng) for _ in range(40000))
                self.assertEqual(set(contagem), validos)
                esperado = 40000 / len(validos)
                for frequencia in contagem.values():
                    self.assertLess(abs(frequencia - esperado), 0.15 * esperado)


class TestCruzamentos(unittest.TestCase):
    PAI1 = [1, 2, 3, 4, 5, 6, 7, 8]
    PAI2 = [8, 6, 4, 2, 7, 5, 3, 1]

    def test_ox_exemplo_numerico(self):
        # Trecho herdado do pai 1 (posicoes 2..4): 3 4 5. O pai 2 lido a
        # partir da posicao 5 e: 5 3 1 8 6 4 2 7; sem 3, 4 e 5 sobra
        # 1 8 6 2 7, que preenche as posicoes 5, 6, 7, 0, 1.
        self.assertEqual(cruzamentos.ox(self.PAI1, self.PAI2, 2, 4), [2, 7, 3, 4, 5, 1, 8, 6])

    def test_pmx_exemplo_numerico(self):
        # Trecho herdado do pai 1 (posicoes 2..4): 3 4 5. No mesmo trecho o
        # pai 2 tem 4 2 7; o 4 ja esta no filho. O 2 (posicao 3) segue o
        # mapeamento: pai1[3] = 4 esta na posicao 2 do pai 2, ainda no
        # trecho; pai1[2] = 3 esta na posicao 6 -> filho[6] = 2. O 7
        # (posicao 4): pai1[4] = 5 esta na posicao 5 -> filho[5] = 7. As
        # posicoes 0, 1 e 7 sao copiadas do pai 2: 8, 6, 1.
        self.assertEqual(cruzamentos.pmx(self.PAI1, self.PAI2, 2, 4), [8, 6, 3, 4, 5, 7, 2, 1])

    def test_filhos_sao_permutacoes(self):
        rng = random.Random(11)
        for _ in range(2000):
            n = rng.randint(1, 30)
            pai1 = rng.sample(range(1, n + 1), n)
            pai2 = rng.sample(range(1, n + 1), n)
            a, b = cruzamentos.sortear_cortes(n, rng)
            for operador in (cruzamentos.ox, cruzamentos.pmx):
                filho = operador(pai1, pai2, a, b)
                self.assertEqual(sorted(filho), list(range(1, n + 1)))
                self.assertEqual(filho[a:b + 1], pai1[a:b + 1])

    def test_pais_iguais_geram_o_mesmo_filho(self):
        rng = random.Random(5)
        pai = rng.sample(range(1, 21), 20)
        for operador in (cruzamentos.ox, cruzamentos.pmx):
            self.assertEqual(operador(pai, pai, 3, 9), pai)


class TestOrcamento(unittest.TestCase):
    def test_para_exatamente_no_orcamento(self):
        monitor = Monitor(orcamento=1000)
        problema = ProblemaManutencao(instancia("pequena"), rng=random.Random(0), monitor=monitor)
        s = problema.solucao_aleatoria()
        with self.assertRaises(OrcamentoEsgotado):
            while True:
                problema.avaliar_sucessor(s, problema.sucessor_aleatorio(s))
        self.assertEqual(monitor.avaliacoes, 1000)

    def test_operadores_do_problema_geram_permutacoes(self):
        problema = ProblemaManutencao(instancia("media"), rng=random.Random(2), cruzamento="pmx")
        for _ in range(500):
            s1 = problema.solucao_aleatoria()
            s2 = problema.solucao_aleatoria()
            for filho in (problema.cruzamento(s1, s2), problema.mutacao(s1)):
                self.assertEqual(sorted(filho), list(range(1, 51)))


if __name__ == "__main__":
    unittest.main()
