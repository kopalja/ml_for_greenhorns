

from diacritization_dictionary import Dictionary

LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"

# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())




d = Dictionary()

v = d.variants['Babicka'][0]

print(v)

#print('ať'.translate(DIA_TO_NODIA) in d.variants)

r = [1, 2, 3, 4, 5, 6]
r2 = [20, 30]
r.extend(r2)

print(r)


word = 'babička'
print(word in d.variants[word.translate(DIA_TO_NODIA)])