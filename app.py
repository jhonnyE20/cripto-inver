from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

model = joblib.load('modelo/cripto-inver.pkl')

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            print(df.columns)
            if 'Nombre' not in df.columns or 'Exchange Principal' not in df.columns:
                return jsonify({"success": False, "message": 'El archivo CSV no contiene las columnas necesarias'})
            
            le_nombre = LabelEncoder()
            df['Nombre_Codificado'] = le_nombre.fit_transform(df['Nombre'])

            le_simbolo = LabelEncoder()
            df['Simbolo_Codificado'] = le_simbolo.fit_transform(df['Símbolo'])

            le_categoria = LabelEncoder()
            df['Categoria_Codificado'] = le_categoria.fit_transform(df['Categoría'])

            le_exchange = LabelEncoder()
            df['Exchange_Codificado'] = le_exchange.fit_transform(df['Exchange Principal'])

            del df['Nombre']
            del df['Símbolo']
            del df['Categoría']
            del df['Exchange Principal']

            cap_min = 1129633.2834955514
            cap_max = 99875044.10451472
            vol_min = 13908.38158095001
            vol_max = 9991792.53054213

            df["Cap_norm"] = (df["Capitalización (USD)"] - cap_min) / \
                       (cap_max - cap_min)

            df["Vol_norm"] = (df["Volumen 24h (USD)"] - vol_min) / \
                       (vol_max - vol_min)

            
            df["Score"] = (1 - df["Cap_norm"]) * 0.6 + df["Vol_norm"] * 0.4

            columnas_excluir = ["Target", "Score", "Cap_norm", "Vol_norm", "Narrativa_Codificada"]

            X = df.drop(columns=[col for col in columnas_excluir if col in df.columns])

            predictions = model.predict(X)

            result = predictions[0]
            
            return jsonify({"success": True, "predicción": result, "valor_predicción": predictions[0]})

        except Exception as e:
            return jsonify({"success": False, "message": str(e)})

    else:
        return jsonify({"success": False, "message": "Por favor, sube un archivo CSV válido"})

if __name__ == '__main__':
    app.run(debug=True)