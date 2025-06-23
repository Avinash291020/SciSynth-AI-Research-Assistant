import os

def list_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".csv")]

def clean_text(text):
    return text.replace("\n", " ").strip() 