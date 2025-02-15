import pandas as pd

def recommend_career(user_skills):
    return ["Data Scientist", "Software Engineer"]

if __name__ == "__main__":
    print(recommend_career(["Python", "Machine Learning"]))
