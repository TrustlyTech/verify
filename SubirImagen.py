from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde .env
load_dotenv()

app = Flask(__name__)

# Configuración de la API de Azure Face usando variables de entorno
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
detect_endpoint = f"{endpoint}/face/v1.0/detect"
identify_endpoint = f"{endpoint}/face/v1.0/identify"
large_person_group_id = "requisitoriadosgroup"  # Asegúrate de que este grupo exista en Azure

# Función para obtener información del `personId` identificado
def get_person_info(person_id):
    person_url = f"{endpoint}/face/v1.0/largepersongroups/{large_person_group_id}/persons/{person_id}"
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    response = requests.get(person_url, headers=headers)
    response.raise_for_status()
    return response.json()

@app.route('/detect_and_identify', methods=['POST'])
def detect_and_identify():
    try:
        # Verificar que la solicitud contenga una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se proporcionó ninguna imagen."}), 400
        
        # Leer la imagen desde los datos enviados
        image_file = request.files['image']
        image_data = image_file.read()

        # Configurar encabezados y parámetros para la detección
        headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-Type": "application/octet-stream"
        }
        detect_params = {
            "returnFaceId": "true",
            "recognitionModel": "recognition_04",
            "detectionModel": "detection_03"
        }

        # Paso 1: Detección de rostro
        detect_response = requests.post(detect_endpoint, headers=headers, params=detect_params, data=image_data)
        detect_response.raise_for_status()

        # Procesar respuesta de detección
        faces = detect_response.json()
        if not faces:
            return jsonify({"message": "No se detectaron rostros en la imagen."}), 404

        face_id = faces[0].get("faceId")
        if not face_id:
            return jsonify({"error": "No se pudo obtener un faceId de la imagen."}), 500

        # Paso 2: Identificación en el grupo
        identify_headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-Type": "application/json"
        }
        identify_body = {
            "faceIds": [face_id],
            "largePersonGroupId": large_person_group_id,
            "maxNumOfCandidatesReturned": 1,
            "confidenceThreshold": 0.6
        }

        # Enviar el faceId al endpoint de identificación
        identify_response = requests.post(identify_endpoint, headers=identify_headers, json=identify_body)
        identify_response.raise_for_status()

        # Procesar la respuesta de identificación
        identify_result = identify_response.json()
        if identify_result and identify_result[0]["candidates"]:
            candidate = identify_result[0]["candidates"][0]
            person_id = candidate["personId"]
            confidence = candidate["confidence"]

            # Obtener información adicional del personId
            person_info = get_person_info(person_id)
            name = person_info.get("name")
            user_data = person_info.get("userData")

            return jsonify({
                "message": "Coincidencia encontrada",
                "personId": person_id,
                "name": name,
                "userData": user_data,
                "confidence": confidence
            }), 200
        else:
            return jsonify({"message": "No se encontraron coincidencias en el grupo."}), 404

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error en la solicitud: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error inesperado: {e}"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

