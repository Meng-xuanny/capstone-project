import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from map import to_numeric


class MedicalChargesGUI:
    def __init__(self, window, data_path='Medical_insurance.csv'):
        self.window = window
        self.window.title('Medical Charges Prediction')

        # Load data
        self.dataNumeric = self.preprocess_data(data_path)

        self.X = self.dataNumeric.drop('charges', axis=1).values
        self.y = self.dataNumeric['charges'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)
        # Load model
        self.model = DecisionTreeRegressor()
        self.model.fit(self.X_train, self.y_train)

        # Create labels and entry fields for input
        self.create_input_widgets()

        # Create the predict button
        self.predict_button = ttk.Button(self.window, text="Predict", command=self.predict_button_click)
        self.predict_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        # Create label to display prediction
        self.prediction_label = ttk.Label(self.window, text="")
        self.prediction_label.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

    @staticmethod
    def preprocess_data(path):
        data = pd.read_csv(path)
        SelectedColumns = ['age', 'bmi', 'children', 'smoker']
        # Selecting final columns
        DataForML = data[SelectedColumns]

        # Define mapping dictionaries for each boolean feature
        smoker_mapping = {'no': 0, 'yes': 1}

        # Map boolean features to numeric values
        dataNumeric = DataForML.copy()
        dataNumeric['smoker'] = dataNumeric['smoker'].map(smoker_mapping)

        # Add Target Variable to the data
        dataNumeric['charges'] = data['charges']

        return dataNumeric

    def create_input_widgets(self):
        labels = ['Age:', 'BMI:', 'Children:', 'Smoker:']
        entries = [ttk.Entry(self.window) for _ in range(3)]
        entries.append(ttk.Combobox(self.window, values=["yes", "no"]))

        for i, (label, entry) in enumerate(zip(labels, entries)):
            ttk.Label(self.window, text=label).grid(row=i, column=0, padx=5, pady=5)
            entry.grid(row=i, column=1, padx=5, pady=5)

        self.age_entry, self.bmi_entry, self.children_entry, self.smoker_combobox = entries

    def predict_charges(self, age, bmi, children, smoker):
        inputs = [age, bmi, children, smoker]
        # Generate predictions
        prediction_result = self.model.predict([inputs])

        return prediction_result

    # Define the button click event
    def predict_button_click(self):
        try:
            # Get input values from the entry fields
            age = float(self.age_entry.get())
            bmi = float(self.bmi_entry.get())
            children = int(self.children_entry.get())
            smoker_value = self.smoker_combobox.get()
            smoker = self.select_smoker(smoker_value)

            # Make prediction using the input values
            prediction = self.predict_charges(age, bmi, children, smoker)

            # Display the prediction
            self.prediction_label.config(text="Predicted medical charges: $%.2f" % prediction)

        except ValueError:
            messagebox.showerror("Error",
                                 "Please enter valid numeric values for age and BMI, and an integer value for children.")

    @staticmethod
    def select_smoker(smoker_value):
        smoker = 0
        if smoker_value == 'yes':
            smoker = 1
        return smoker


if __name__ == '__main__':
    root = tk.Tk()
    app = MedicalChargesGUI(root)
    root.mainloop()
