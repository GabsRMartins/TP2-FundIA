"""T1: a funcao de avaliacao reproduz os valores de verificacao do enunciado."""

import itertools
import unittest

from tests.comum import instancia

from utils.referencia import guloso_menor_acrescimo, otimo_exato


class TestExemploFigura1(unittest.TestCase):
    """Instancia n = 4 da Figura 1 (coordenadas deduzidas do cronograma)."""

    def setUp(self):
        self.inst = instancia("exemplo")

    def test_tempos_de_deslocamento(self):
        t = self.inst.t
        self.assertEqual((t[0][1], t[1][4], t[4][2], t[2][3]), (5, 5, 8, 10))

    def test_rota_do_guloso_por_proximidade_custa_39(self):
        self.assertEqual(self.inst.atraso_ponderado([1, 4, 2, 3]), 39)
        conclusoes = [c for _, c, _ in self.inst.cronograma([1, 4, 2, 3])]
        self.assertEqual(conclusoes, [7, 16, 27, 38])

    def test_otimo_custa_22_por_forca_bruta(self):
        custo, ordem = min(
            (self.inst.atraso_ponderado(perm), list(perm))
            for perm in itertools.permutations(range(1, 5))
        )
        self.assertEqual((custo, ordem), (22, [1, 4, 3, 2]))


class TestInstanciaPequena(unittest.TestCase):
    def setUp(self):
        self.inst = instancia("pequena")

    def test_tamanho_e_soma_das_duracoes(self):
        self.assertEqual(self.inst.n, 12)
        self.assertEqual(sum(self.inst.p), 55)

    def test_identidade_vale_4366(self):
        self.assertEqual(self.inst.atraso_ponderado(range(1, 13)), 4366)

    def test_guloso_de_referencia_vale_3168(self):
        custo, ordem = guloso_menor_acrescimo(self.inst)
        self.assertEqual(custo, 3168)
        self.assertEqual(self.inst.atraso_ponderado(ordem), 3168)

    def test_otimo_exato_vale_1875(self):
        custo, ordem = otimo_exato(self.inst)
        self.assertEqual(custo, 1875)
        self.assertEqual(sorted(ordem), list(range(1, 13)))
        self.assertEqual(self.inst.atraso_ponderado(ordem), 1875)


class TestDemaisInstancias(unittest.TestCase):
    def test_tamanhos_e_somas_das_duracoes(self):
        for nome, n, soma in (("media", 50, 239), ("grande", 150, 684)):
            with self.subTest(nome=nome):
                inst = instancia(nome)
                self.assertEqual((inst.n, sum(inst.p)), (n, soma))


if __name__ == "__main__":
    unittest.main()
