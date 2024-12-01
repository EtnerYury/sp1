import cv2
import numpy as np
import onnxruntime as rt
from rembg import remove
from PIL import Image
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from rembg import new_session

app = FastAPI()
# Crear una sesión de rembg una sola vez
session = new_session()

def process_image_with_white_background(input_array):
    """
    Procesa una imagen (numpy array) eliminando el fondo y reemplazándolo con blanco
    """
    input_pil = Image.fromarray(cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB))
    
    # Quitar el fondo usando la sesión existente
    output_image = remove(input_pil, session=session)
    
    # Crear una nueva imagen con fondo blanco
    background = Image.new("RGBA", output_image.size, (255, 255, 255, 255))
    
    # Combinar la imagen sin fondo con el fondo blanco
    combined_image = Image.alpha_composite(background, output_image)
    
    # Convertir a RGB y luego a array BGR
    final_array = cv2.cvtColor(np.array(combined_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    return final_array

def extract_features(image_path, fixed_size=(350, 450)):
    """
    Extrae características de la imagen
    """
    try:
        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] No se pudo leer la imagen: {image_path}")
            return None
            
        # Redimensionar
        image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)
        image = process_image_with_white_background(image)
        
        # Extraer características
        def fd_histogram(image, bins=8):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()

        def fd_haralick(image):
            import mahotas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            haralick = mahotas.features.haralick(gray).mean(axis=0)
            return haralick

        def fd_hu_moments(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = cv2.HuMoments(cv2.moments(image)).flatten()
            return feature

        # Extraer todas las características
        hist_features = fd_histogram(image)
        haralick_features = fd_haralick(image)
        hu_features = fd_hu_moments(image)

        # Concatenar todas las características
        global_feature = np.hstack([hist_features, haralick_features, hu_features])
        
        return global_feature.reshape(1, -1).astype(np.float32)
        
    except Exception as e:
        print(f"[ERROR] Error en extracción de características: {str(e)}")
        return None

def predict_with_onnx(features, model_path):
    """
    Realiza la predicción usando el modelo ONNX
    """
    try:
        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        
        prediction = sess.run([label_name], {input_name: features})[0]
        
        return int(prediction[0]), 1.0  # Retornamos la predicción y una confianza de 1.0
    except Exception as e:
        print(f"[ERROR] Error en predicción ONNX: {str(e)}")
        return None, None

@app.get("/health/")
async def health():
    return {"status": "ok"}

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    try:
        # Ruta para el archivo temporal
        temp_file_path = f"temp_{file.filename}"
        # Ruta del modelo
        model_path = "random_forest_model.onnx"
        
        print(f"[INFO] Procesando archivo: {file.filename}")
        
        try:
            # Guardar archivo
            with open(temp_file_path, "wb") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
            print("[INFO] Archivo guardado temporalmente")
            
            # Verificar modelo
            if not os.path.exists(model_path):
                print(f"[ERROR] Modelo no encontrado: {model_path}")
                return JSONResponse(
                    content={"error": f"Model not found at {model_path}"},
                    status_code=404
                )
            
            # Extraer características
            features = extract_features(temp_file_path)
            if features is None:
                print("[ERROR] Fallo en extracción de características")
                return JSONResponse(
                    content={"error": "Feature extraction failed"},
                    status_code=400
                )
            
            # Predicción
            predicted_class, confidence = predict_with_onnx(features, model_path)
            
            if predicted_class is None:
                print("[ERROR] La predicción retornó None")
                return JSONResponse(
                    content={"error": "Prediction failed"},
                    status_code=400
                )
            
            print(f"[INFO] Predicción exitosa: clase={predicted_class}, confianza={confidence}")
            return {
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": confidence
            }
                
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print("[INFO] Archivo temporal eliminado")
                
    except Exception as e:
        print(f"[ERROR] Error inesperado: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

