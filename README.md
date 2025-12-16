# 🤟 Static Sign Language Translator (Project Documentation)

## 📌 Project Overview

This project is a **Static Sign Language Translator** that recognizes **non-moving (static) hand signs** using computer vision and machine learning. The system captures hand landmarks using **MediaPipe**, extracts features, and classifies the sign using a trained ML model.

The goal of this project is to help **deaf and mute individuals** communicate basic needs and emotions using simple, clearly defined static hand gestures.

---

## 🎯 Key Features

* Supports **static hand signs only** (no motion-based gestures)
* Real-time detection using webcam
* Uses **MediaPipe Hands (21 landmarks)**
* Machine Learning model for classification
* Easily extendable with new signs

---

## 🧠 Technology Stack

* **Python**
* **OpenCV** – webcam input
* **MediaPipe** – hand landmark detection
* **Scikit-learn** – model training (RandomForest)
* **NumPy / Pandas** – data handling

---

## ✋ Supported Static Signs

Below are the signs currently included in the model. Each sign uses a **single fixed hand pose**.

---

### 🧑 Name

**Purpose:** Asking or telling someone their name

![Name Sign](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/ASL-sign-name.svg/512px-ASL-sign-name.svg.png)

---

### 🏠 Address

**Purpose:** Used when asking for or sharing address information

![Address Sign](https://www.signingsavvy.com/images/words/address.jpg)

---

### 🚨 Emergency

**Purpose:** Indicates an emergency situation

![Emergency Sign](https://www.signingsavvy.com/images/words/emergency.jpg)

---

### 👍 Yes

**Purpose:** Positive confirmation

![Yes Sign](https://www.signingsavvy.com/images/words/yes.jpg)

---

### 👎 No

**Purpose:** Negative response

![No Sign](https://www.signingsavvy.com/images/words/no.jpg)

---

### 👋 Hi

**Purpose:** Greeting (static version only)

![Hi Sign](https://www.signingsavvy.com/images/words/hi.jpg)

---

### 🥤 Drink

**Purpose:** Asking for water or a drink

![Drink Sign](https://www.signingsavvy.com/images/words/drink.jpg)

---

### ❤️ I Love You

**Purpose:** Expressing love or care

![I Love You Sign](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/ILY_ASL.svg/512px-ILY_ASL.svg.png)

---



## 📈 Future Improvements

* Add more static signs (doctor, pain, help, stop)
* Improve accuracy with more training samples
* Add text-to-speech output
* Deploy as a web or mobile application

---



## ✅ Note

This project intentionally avoids dynamic gestures to maintain **high accuracy** and **simplicity** for real-time recognition.
