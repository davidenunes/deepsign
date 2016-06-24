import unittest
import deepsign.text.windows as txtw


class TestSlidingWindow(unittest.TestCase):
    def test_window(self):
        sentence ="hello there mr smith welcome back to the world"
        print(sentence)
        tokens = sentence.split(" ")

        windows = txtw.sliding_windows(tokens, window_size=3)

        for w in windows:
            print(w)

        self.assertEqual(len(windows),len(tokens))

if __name__ == '__main__':
    unittest.main()