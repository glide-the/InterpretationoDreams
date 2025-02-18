def save_questions(questions, path):
    with open(path, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(question + "\n")


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        for line in f:
            questions.append(line.strip())
    return questions
