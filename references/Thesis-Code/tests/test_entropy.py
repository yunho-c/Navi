import numpy as np

import entropy


def test_entropy_single_color():
    img = np.ones((100, 100), np.uint8) * 127
    hist = entropy.gradient_histogram(img)
    ent = entropy.entropy(hist)
    assert ent == 0


def test_entropy_gradient_is_small():
    img = np.zeros((100, 100), np.uint8)
    for i in range(100):
        img[i, :] = i

    hist = entropy.gradient_histogram(img)
    ent = entropy.entropy(hist)
    assert ent < 1


def test_entropy_uniform_is_high():
    img = np.uint8(np.random.uniform(0, 255, (100, 100)))

    hist = entropy.gradient_histogram(img)
    ent = entropy.entropy(hist)
    assert ent > 10