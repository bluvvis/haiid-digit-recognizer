from recognition.model import load_and_train_model, print_model_info
from recognition.image_utils import select_and_process_image
from recognition.explainability import explain_prediction
from recognition.feedback import get_user_feedback, log_feedback

def main():
    print("Loading and training the model...")
    print("This is a research system. Your answers will help make AI better!")
    
    model = load_and_train_model()
    print_model_info(model)  
    while True:
        print("\n1. Upload an image and recognize it")
        print("2. Exit")
        choice = input("Select an action: ")

        if choice == '1':
            image_path = input("Enter the path to the image: ")
            processed = select_and_process_image(image_path)
            if processed is not None:
                prediction = model.predict(processed)
                proba = model.predict_proba(processed)[0]
                print(f"Модель предсказала: {prediction[0]}")
                explain_prediction(model, processed)
                feedback = get_user_feedback(prediction[0])
                log_feedback(image_path, prediction[0], feedback)
        elif choice == '2':
            print("See you soon!")
            break
        else:
            print("Invalid input.")

if __name__ == "__main__":
    main()
