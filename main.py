from recognition.model import load_and_train_model, print_model_info
from recognition.image_utils import select_and_process_image
from recognition.explainability import explain_prediction
from recognition.feedback import get_user_feedback, log_feedback

def main():
    print("Загрузка и обучение модели...")
    print("Это исследовательская система. Ваши ответы помогут сделать ИИ лучше!")
    
    model = load_and_train_model()
    print_model_info(model)  
    while True:
        print("\n1. Загрузить изображение и распознать")
        print("2. Выход")
        choice = input("Выберите действие: ")

        if choice == '1':
            image_path = input("Введите путь к изображению: ")
            processed = select_and_process_image(image_path)
            if processed is not None:
                prediction = model.predict(processed)
                proba = model.predict_proba(processed)[0]
                print(f"Модель предсказала: {prediction[0]}")
                explain_prediction(model, processed)
                feedback = get_user_feedback(prediction[0])
                log_feedback(image_path, prediction[0], feedback)
        elif choice == '2':
            print("До встречи!")
            break
        else:
            print("Неверный ввод.")

if __name__ == "__main__":
    main()
