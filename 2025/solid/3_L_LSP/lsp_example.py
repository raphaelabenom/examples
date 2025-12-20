class Animal:
    def comer(self):
        print("O Animal comendo")

    def andar(self):
        print("O animal está andando na jaula")


class Felino(Animal):
    def lamber(self):
        print("O Felino está lambendo seu pelo")


class Leao(Felino):
    def rugir(self):
        print("O leao esta rugindo alto !!!")


class Pessoa:
    def observa(self, animal: Animal):
        animal.comer()


renatinho = Pessoa()
animal = Animal()
felino = Felino()
leao = Leao()

renatinho.observa(leao)
