
class TOEFLReader:

    def __init__(self,questions_file, answers_file):
        """

        :param question_file: a file handle for the questions file
        :param answer_file: a file handle for the answers file
        """
        self.words = set()
        self.questions = []
        self.answers = []

        question_it = iter(questions_file)
        answer_it = iter(answers_file)

        while len(self.questions) < 80:
            # read question and options
            line = next(question_it)
            (c, w) = line.split()
            c = int(c)

            # read 4 options
            answers = [next(question_it).split()[1] for _ in range(4)]
            self.words.add(w)
            self.words.update(answers)

            question = (
                w,
                answers
            )
            self.questions.append(question)

            # read answer for current question
            line = next(answer_it)
            (_,a) = line.split()
            a = ord(a) - 97
            self.answers.append(a)

    def answer(self,question_index):
        return self.answers[question_index]















