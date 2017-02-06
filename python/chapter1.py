import collections
from math import hypot

'''
Card = collections.namedtuple('Card', ['rank', 'suit'])

class FD :
    ranks = [str(n) for n in range(2, 11) ] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
    def __len__(self):
        return (len(self._cards))
    def __getitem__(self, position):
        return self._cards[position]

beer_card = Card('7', 'diamonds')

print(beer_card)

aaa = FD()

print ( len(aaa) )

print (aaa[1])

for a in aaa :
    print (a)
'''


class Vector :
    def __init__(self , x=0, y=0):
        self.x = x
        self.y = y

v1 = Vector(3, 4)




