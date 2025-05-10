import csv
import os

def log_feedback(image_name, predicted, correct_digit):
    """Логирует обратную связь в CSV."""
    os.makedirs("data", exist_ok=True)
    file_path = "data/feedback_log.csv"
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["image_name", "predicted_digit", "correct_digit"])
        writer.writerow([image_name, predicted, correct_digit])
