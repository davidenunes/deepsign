import unittest
import deepsign.text.transform as transf


class TestTransform(unittest.TestCase):

    def testCaseTransform(self):
        case = transf.CaseTransform()

        s1 = "Hello there Mr Smith welcome back to the World"

        s2 = case.apply(s1.split(" "))

    def testCleanPunctuation(self):
        punct = transf.CleanPunctuation()

        s1 = "Hello there Mr Smith , welcome back to the World !"
        s1a = s1.split(" ")

        s2 = punct.apply(s1a)
        self.assertEqual(len(s1a)-2,len(s2))

    def testTransformPipeline(self):
        pipe = transf.TransformPipeline()
        pipe.append(transf.CleanPunctuation())
        pipe.append(transf.CaseTransform())

        case = transf.CaseTransform()

        s1 = "Hello there Mr Smith welcome back to the World"
        s2 = "Hello there Mr Smith , welcome back to the World !"

        res1 = case.apply(s1.split(" "))
        res2 = pipe.apply(s2.split(" "))

        self.assertListEqual(res1,res2)


if __name__ == '__main__':
    unittest.main()