import unittest

import numpy as np

from detector import ClipBuffer


class MyTestCase(unittest.TestCase):
    def test_something(self):
        buffer = ClipBuffer(1, 4)
        motion = [1,0,1,1,0,1,0,0]
        windows = [3,2,3,2,1]
        for m in motion:
            buffer.append(np.empty(0), m)


        self.assertListEqual(windows, list(buffer.window_totals))

    def test_trim(self):
        buffer = ClipBuffer(1, 4)
        motion = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]
        windows = [3, 2, 3, 2, 1, 1, 0]
        for m in motion:
            buffer.append(np.empty(0), m)
        self.assertListEqual(windows, list(buffer.window_totals))

if __name__ == '__main__':
    unittest.main()
