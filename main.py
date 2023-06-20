import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


class GlassClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Glass Classifier")

        window_width = 400
        window_height = 400
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 14))
        style.configure("TLabel", font=("Arial", 14))

        self.btn_load_data = ttk.Button(root, text="Wczytaj dane", command=self.load_data)
        self.btn_load_data.pack(pady=10)

        self.btn_build_model = ttk.Button(root, text="Zbuduj model", command=self.build_model)
        self.btn_build_model.pack(pady=10)

        self.btn_test_model = ttk.Button(root, text="Testuj model", command=self.test_model, state=tk.DISABLED)
        self.btn_test_model.pack(pady=10)

        self.btn_predict_label = ttk.Button(root, text="Przewiduj etykietę", command=self.predict_label, state=tk.DISABLED)
        self.btn_predict_label.pack(pady=10)

        self.btn_add_data = ttk.Button(root, text="Dodaj dane", command=self.add_data)
        self.btn_add_data.pack(pady=10)

        self.btn_save_model = ttk.Button(root, text="Zapisz model", command=self.save_model)
        self.btn_save_model.pack(pady=10)

        self.btn_load_model = ttk.Button(root, text="Wczytaj model", command=self.load_model)
        self.btn_load_model.pack(pady=10)

        self.model = None

        self.data = pd.DataFrame()
        self.X = pd.DataFrame()
        self.y = pd.Series()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=(("data files", "*.data"), ("All files", "*.*")))
        if file_path:
            try:
                self.data = pd.read_csv(file_path, header=None)
                self.btn_test_model.config(state=tk.NORMAL)
                self.btn_predict_label.config(state=tk.NORMAL)
                self.X = self.data.iloc[:, :-1]
                self.y = self.data.iloc[:, -1]
                messagebox.showinfo("Sukces", "Dane zostały wczytane.")
            except:
                messagebox.showerror("Błąd", "Wystąpił błąd podczas wczytywania danych.")

    def build_model(self):
        if not self.X.empty and not self.y.empty:
            self.model = LogisticRegression(max_iter=1000) #self.model = LogisticRegression(solver='liblinear')

            self.model.fit(self.X, self.y)

            messagebox.showinfo("Sukces", "Model został zbudowany.")
            self.btn_test_model.config(state=tk.NORMAL)
            self.btn_predict_label.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Błąd", "Brak danych do zbudowania modelu.")

    def test_model(self):
        if self.model is not None and not self.X.empty and not self.y.empty:
            accuracy = self.model.score(self.X, self.y)

            messagebox.showinfo("Wynik testowania", f"Dokładność modelu: {accuracy:.2%}")
        else:
            messagebox.showerror("Błąd", "Brak zbudowanego modelu lub danych do testowania.")

    def predict_label(self):
        if self.model is not None and not self.X.empty:
            input_data = simpledialog.askstring("Przewidywanie etykiety", "Wprowadź dane oddzielone przecinkami:")

            if input_data:
                try:
                    input_data = [float(x.strip()) for x in input_data.split(",")]

                    # Tworzenie DataFrame z pojedynczym wierszem danych wejściowych
                    input_df = pd.DataFrame([input_data], columns=self.X.columns)

                    # Przewidywanie etykiety
                    predicted_label = self.model.predict(input_df)[0]

                    messagebox.showinfo("Przewidywana etykieta", f"Etykieta przewidziana dla danych: {predicted_label}")
                except:
                    messagebox.showerror("Błąd", "Wystąpił błąd podczas przetwarzania danych.")
        else:
            messagebox.showerror("Błąd", "Brak zbudowanego modelu lub danych do przewidzenia.")

    def add_data(self):
        if not self.data.empty:
            input_data = simpledialog.askstring("Dodawanie danych", "Wprowadź nowe dane oddzielone przecinkami:")

            try:
                input_data = [float(x.strip()) for x in input_data.split(",")]

                new_data = pd.DataFrame([input_data], columns=self.data.columns)
                self.data = pd.concat([self.data, new_data], ignore_index=True)
                self.X = self.data.iloc[:, :-1]
                self.y = self.data.iloc[:, -1]

                messagebox.showinfo("Sukces", "Nowe dane zostały dodane.")
            except:
                messagebox.showerror("Błąd", "Wystąpił błąd podczas dodawania nowych danych.")
        else:
            messagebox.showerror("Błąd", "Brak wczytanych danych.")

    def save_model(self):
        if self.model is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".data",
                                                     filetypes=(("Data files", "*.data"), ("All files", "*.*")))

            try:
                joblib.dump(self.model, file_path)

                messagebox.showinfo("Sukces", "Model został zapisany.")
            except:
                messagebox.showerror("Błąd", "Wystąpił błąd podczas zapisywania modelu.")
        else:
            messagebox.showerror("Błąd", "Brak zbudowanego modelu.")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=(("Data files", "*.data"), ("All files", "*.*")))
        if file_path:
            try:
                with open(file_path, "rb") as file:
                    self.model = pickle.load(file)
                messagebox.showinfo("Sukces", "Model został wczytany.")
                self.X = self.data.iloc[:, :-1]
                self.y = self.data.iloc[:, -1]
                self.btn_test_model.config(state=tk.NORMAL)
                self.btn_predict_label.config(state=tk.NORMAL)
            except:
                messagebox.showerror("Błąd", "Wystąpił błąd podczas wczytywania modelu.")


root = tk.Tk()
app = GlassClassifierGUI(root)
root.mainloop()
