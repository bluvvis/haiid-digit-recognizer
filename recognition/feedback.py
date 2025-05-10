import csv
import os

def get_user_feedback(predicted):
    correct = input(f"Это правильно? (y/n): ")
    if correct.lower() == 'y':
        return predicted
    else:
        correct_label = input("Введите правильную цифру: ")
        return correct_label

def log_feedback(image_path, predicted, user_label):
    os.makedirs("data", exist_ok=True)
    with open("data/feedback_log.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_path, predicted, user_label])

