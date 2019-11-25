

from diacritization_dictionary import Dictionary

LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"

# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

import numpy as np


d = Dictionary()




print(d.variants['premozitel'])

exit()

s = 0
for x in d.variants.keys():
    print(d.variants[x])
    s += len(d.variants[x])

print(s / len(d.variants.keys()))


r = [1, 2, 3, 4, 5, 6]
r2 = [20, 30]
r.extend(r2)

exit()

print(r)


print(word in d.variants[word.translate(DIA_TO_NODIA)])