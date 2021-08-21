import numpy as np
from mcholmz.mcholmz import modifiedChol as modChol

if __name__ == '__main__':
    G = np.array([[2, 6, 10],
                  [6, 10, 14],
                  [10, 14, 18]], dtype=np.float64)

    L, d, e = modChol(G)
    Err = (L @ np.diag(d.flatten()) @ L.T) - G
    print('Difference between original and decomposed matrices:\n', Err)
