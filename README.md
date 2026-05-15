This is a personal project utilizing AI to make a simulation using a fun problem called the Secretary Problem

The Paper itself is titled "An Empirical Investigation of Secretary-Inspired Stopping Rules for Neural Networks" -still pending

What it does:
Treats each training epoch as a candidate in the Secretary Problem. Observe the first K% of epochs, then stop at the first epoch that beats the observed best — no patience parameter needed.

Methods:
Full Training (100 epochs)
Early Stopping (patience = 5)
Secretary 37%, 50%, 75%

Usage [i used WSL]:
bashpip install tensorflow numpy pandas scikit-learn tqdm
python3 main.py

Results are saved to results/.

Requirements:
Python 3.x, TensorFlow 2.11+
