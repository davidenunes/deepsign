import unittest
import deepsign.text.transform as transf


class TestTransform(unittest.TestCase):

    def testCaseTransform(self):

        s1 = "Hello there Mr Smith welcome back to the World"

        s2 = transf.to_lower(s1.split(" "))

        print(s2)

    def testCleanPunctuation(self):

        s1 = "Hello there Mr Smith , welcome back to the World !"
        s1a = s1.split(" ")

        s2 = transf.rm_punctuation(s1a)
        self.assertEqual(len(s1a)-2,len(s2))

        print(s2)


if __name__ == '__main__':
    unittest.main()