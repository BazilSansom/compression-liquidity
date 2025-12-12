import numpy as np

def bardoscia_cycle_V(pbar: np.ndarray) -> np.ndarray:
    Pi = np.array([
        [0.0, 1.0, 0.0],  # A -> B
        [0.0, 0.0, 1.0],  # B -> C
        [1.0, 0.0, 0.0],  # C -> A
    ])
    return np.diag(pbar) @ Pi

def bardoscia_example1():
    V = bardoscia_cycle_V(np.array([10.0, 7.0, 5.0]))
    e0 = np.array([12.0, 4.0, 3.0])
    return V, e0

def bardoscia_example2():
    V = bardoscia_cycle_V(np.array([10.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])
    return V, e0

def bardoscia_example3():
    V = bardoscia_cycle_V(np.array([2.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])
    return V, e0
