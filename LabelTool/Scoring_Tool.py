import tkinter as tk
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from tkinter import PhotoImage
from Elo import Elo

class ScoringTool:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Scoring Tool")
        self.window.geometry("800x500")
        self.window.resizable(0, 0)
        # self.window.wm_attributes("-transparentcolor", "white")
        
        self.transparent_image = PhotoImage(width=1, height=1)

        self.file_list = self.get_file_list()
        self.image_combinations = self.image_combination()
        self.image1, self.image2 = self.get_image()
        self.record = pd.read_csv("./LabelTool/record.csv", sep=",")

        self.canvas = tk.Canvas(self.window, highlightthickness=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.score_label = tk.Label(
            self.window, text="Which one is better?", font=("Arial", 20), bg="white", fg="black")
        self.score_label.place(x=270, y=40)

        self.score_label1 = tk.Label(
            self.window, text="Click on the Image if you like it !!!", font=("Arial", 15))
        self.score_label1.place(x=255, y=100)

        self.image1_button = tk.Button(self.window, image=self.image1, command=lambda: self.scoring(1))
        self.image1_button.place(x=90, y=150)
        self.image1_button.lift()

        self.image2_button = tk.Button(self.window, image=self.image2, command=lambda: self.scoring(2))
        self.image2_button.place(x=450, y=150)
        self.image2_button.lift()

        image_path = "./LabelTool/Roasted_coffee_beans.jpg"
        self.set_background_with_opacity(image_path)

        self.window.mainloop()

    def get_file_list(self):
        file_list = os.listdir("./LabelTool/ForTestingImage")
        file_list.sort(key=lambda x: int(x.split(".")[0]))
        return file_list

    def image_combination(self):
        temp = []
        for i in range(len(self.file_list)):
            for j in range(i + 1, len(self.file_list)):
                temp.append([i, j])
        return temp

    def get_image(self):
        if not self.image_combinations:
            return None, None

        random_index = np.random.randint(0, len(self.image_combinations))
        self.image1_index, self.image2_index = self.image_combinations[random_index]
        self.image_combinations.pop(random_index)

        image1 = Image.open("./LabelTool/ForTestingImage/" +
                            self.file_list[self.image1_index])
        image1 = image1.resize((250, 250))
        image1 = ImageTk.PhotoImage(image1)

        image2 = Image.open("./LabelTool/ForTestingImage/" +
                            self.file_list[self.image2_index])
        image2 = image2.resize((250, 250))
        image2 = ImageTk.PhotoImage(image2)
        
        print("Picture 1: " + self.file_list[self.image1_index], ", Picture 2: " + self.file_list[self.image2_index])

        return image1, image2

    def scoring(self, button_pressed):
        if button_pressed == 1:
            print("Image 1 button pressed")
            write = [self.image1_index, self.image2_index]
            Elo(self.image1_index, self.image2_index)
            
        elif button_pressed == 2:
            print("Image 2 button pressed")
            write = [self.image2_index, self.image1_index]
            Elo(self.image2_index, self.image1_index)

        self.record = pd.concat([self.record, pd.DataFrame([write], columns=["WinnerID", "LoserID"])], ignore_index=True)
        self.record.to_csv("./LabelTool/record.csv", index=False)
        
        new_image1, new_image2 = self.get_image()

        if new_image1 is not None and new_image2 is not None:
            self.image1 = new_image1
            self.image2 = new_image2
            self.image1_button.configure(image=self.image1)
            self.image2_button.configure(image=self.image2)
        else:
            self.score_label.configure(text="No more images to score")
            self.image1_button.configure(state=tk.DISABLED)
            self.image2_button.configure(state=tk.DISABLED)
            
            
    def set_background_with_opacity(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGBA")
        transparent_image = Image.new("RGBA", image.size, (255, 255, 255, 128))
        blended_image = Image.alpha_composite(image, transparent_image)
        blended_image = blended_image.convert("RGB")
        photo = ImageTk.PhotoImage(blended_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

# 重置所有評分紀錄及圖片分數
RESET = True

if __name__ == "__main__":
    
    if RESET == True:
        record = pd.DataFrame(columns=["WinnerID", "LoserID"])
        record.to_csv("./LabelTool/record.csv", index=False)
        
        df = pd.read_csv("./LabelTool/ForTestingImage.csv")
        imgScore = df.iloc[:, 1].values
        for i in range(len(imgScore)):
            imgScore[i] = 1500
        df.iloc[:, 1] = imgScore
        df.to_csv("./LabelTool/ForTestingImage.csv", index=False)
        
    ScoringTool()
