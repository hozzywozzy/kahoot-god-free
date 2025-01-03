import keyboard
import pyautogui
import pytesseract
import logging
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO)

# Path to tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# you'll need to sign up and use a token for huggingface
huggingface_token = 'YOUR_HUGGINGFACE_API_TOKEN'
hf_model_url = "https://api-inference.huggingface.co/models/gpt2"  # random model


def preprocess_image(image):

    img_array = np.array(image)

    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    thresholded = cv2.adaptiveThreshold(
        enhanced_img[:, :, 0], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return thresholded

def extract_text_from_image(region_coords):
    x1, y1, x2, y2 = region_coords
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    preprocessed_image = preprocess_image(screenshot)
    
    text = pytesseract.image_to_string(preprocessed_image).strip()
    return text

# calls the HF api for the answer (you can change the model to another one on hf)
def huggingface_answer(question_and_answers):
    headers = {"Authorization": f"Bearer {huggingface_token}"}
    payload = {"inputs": question_and_answers}

    response = requests.post(hf_model_url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            answer = response.json()[0]['generated_text'].strip()
            logging.info(f"Hugging Face response: {answer}")
            return int(answer)  
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return None
    else:
        logging.error(f"Hugging Face API error: {response.status_code}")
        return None

def click_button(button):
    button_coords = {1: (0.23, 0.65), 2: (0.77, 0.65), 3: (0.20, 0.80), 4: (0.87, 0.80)}
    screen_width, screen_height = pyautogui.size()
    x, y = button_coords[button]
    x = int(screen_width * x)
    y = int(screen_height * y)
    pyautogui.click(x, y)

def kahoot_god():

    question_and_answers_coords = {
        0: (0.0, 0.072, 1.0, 0.2),  # Question 
        1: (0.042, 0.667, 0.458, 0.742),  #  option 1
        2: (0.542, 0.667, 1.0, 0.742),  #  option 2
        3: (0.042, 0.8, 0.458, 0.875),   #  option 3
        4: (0.542, 0.8, 1.0, 0.875),    #  option 4
    }

    screen_width, screen_height = pyautogui.size()
    res = ""
    
    # Adjust coordinates for any screen resolution
    for element in question_and_answers_coords:
        coords = question_and_answers_coords[element]
        text = extract_text_from_image(coords)
        
        if not text:
            logging.warning(f"No text detected for element {element}. Skipping.")
            continue
        
        if element == 0:
            res += f"Question: {text}\n"
        else:
            res += f"{element}: {text}\n"
    
    logging.info("Extracted Question and Answers:\n" + res)
    
    try:
        answer = huggingface_answer(res)
        if answer:
            click_button(answer)
            logging.info(f"Clicked button: {answer}")
        else:
            logging.error("Failed to get a valid answer.")
    except Exception as e:
        logging.error(f"Error in kahoot_god: {e}")

# Bind the function to a hotke
keyboard.add_hotkey("ctrl+alt+t", kahoot_god)

keyboard.wait()
