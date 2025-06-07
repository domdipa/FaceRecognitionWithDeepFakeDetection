from fastapi import FastAPI, UploadFile, Form, File, Body
from typing import Annotated
from deepface import DeepFace
import mediapipe as mp
import cv2
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import DeepFakeDetection.models as models
from openai import OpenAI
import json
import base64
from enum import Enum
import numpy as np

app = FastAPI()

# Initialize Mediapipe's Face Mesh model for landmark detection
mp_face_mesh = mp.solutions.face_mesh

@app.post("/faceDetection/")
async def check_faces(file1: UploadFile, file2: UploadFile) -> models.FaceDetectionModel:
    # Read uploaded images into memory
    file1_content = await file1.read()
    file2_content = await file2.read()
    
    file1_image = bytes_to_image(file1_content)
    file2_image = bytes_to_image(file2_content)
        
    deepface_models = [
        "VGG-Face", 
        "Facenet512", 
        "GhostFaceNet"
    ]
    
    deepface_result_list = []
    
    # Run DeepFace verification to check if the two images are of the same person
    for model in deepface_models:
        try:
            result = DeepFace.verify(file1_image, file2_image, model_name=model)

            deepface_verification_model = models.DeepFaceVerificationModel(
                model=result["model"],
                deepface_verify= result["verified"],
                threshold=result["threshold"],
                cosine_distance= result["distance"],
                detector_backend=result["detector_backend"]            
            )
            
            deepface_result_list.append(deepface_verification_model)
        except Exception as e:
            print('Error in deepFaceVerification')
    
    # Combine results into a structured response model
    output = models.FaceDetectionModel(
        deepface_verification_list=deepface_result_list
    )
    
    landmarks1, landmarks2 = get_landmarks(file1_image, file2_image)
    if landmarks1 != None or landmarks2 != None:
        output.face1_face_landmarks = landmarks1
        output.face2_face_landmarks = landmarks2

    # Return results as a JSON response
    json_compatible_item_data = jsonable_encoder(output)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/checkByLLM/")
async def verify_by_lmm(faceDetectionModel: str = Form(...),
                        file1: UploadFile = File(...),
                        file2: UploadFile = File(...)) -> models.LLMResultModel:
    file1_content = await file1.read()
    file2_content = await file2.read()
    
    file1_image = bytes_to_image(file1_content)
    file2_image = bytes_to_image(file2_content)
    
    face1_cropped = crop_face(file1_image)
    face2_cropped = crop_face(file2_image)
    
    base64_image_original = encode_image(face1_cropped)
    base64_image_second = encode_image(face2_cropped)
    
    prompt = f"""
    Context: You are an advanced AI specializing in face verification and deepfake detection.  
    Your task: Determine if two images belong to the same person or if the second image is a deepfake.  

    Input Data:
    - Two images for direct visual analysis.
    - A JSON input with the following structure and example values:
    {{
        "deepface_verification_list": [
        {{
            "model": "VGG-Face",
            "deepface_verify": false,
            "threshold": 0.68,
            "cosine_distance": 0.6858041872665477,
            "detector_backend": "opencv"
        }},
        {{
            "model": "Facenet512",
            "deepface_verify": false,
            "threshold": 0.3,
            "cosine_distance": 0.46313974036310845,
            "detector_backend": "opencv"
        }},
        {{
            "model": "GhostFaceNet",
            "deepface_verify": true,
            "threshold": 0.65,
            "cosine_distance": 0.4348621277132364,
            "detector_backend": "opencv"
        }}
        ],
        "face1_face_landmarks": 
        {{
            "image_shape": "128x128",
            // Various facial landmark arrays (e.g., "silhouette", "lipsUpperOuter", etc.)
            // representing key regions such as the face outline, lips, eyes, nose, and cheeks.
            ...
        }},
        "face2_face_landmarks": 
        {{
            "image_shape": "128x128",
            // Similar landmark arrays as for face2.
            ...
        }}
    }}

    Task Instructions:
    1. DeepFace Verification: Evaluate the outputs from the deepface_verification_list. Dynamically adjust the weight of each model based on its performance, especially when the results are ambiguous (e.g., mixed or partial matches).
    2. Landmark Comparison: Normalize the landmark coordinates using the provided "image_shape" for each face. Compare the corresponding facial landmarks (e.g., from the silhouette, lips, eyes, and nose) between face1 and face2. Pay special attention to asymmetries, proportional deviations, or other inconsistencies that may indicate tampering.
    3. Visual Analysis: Examine the two images for visual inconsistencies (such as lighting differences, blurring, or other artifacts) that could affect landmark detection and model performance.
    4. Integrative Decision Making: Combine the insights from DeepFace verification, landmark comparison and visual analysis. If the results are contradictory, give possible reasons.
    5. Final Output:
    Return a final decision along with:
        - "verified": true/false indicating if the two images belong to the same person.
        - "similarity": a value between 0.0 and 1.0 representing the similarity between the images with a threshold set to 0.8.
        - "confidence_score": a value between 0.0 and 1.0 expressing your overall confidence in the result.
        - "explanation": a short justification for your decision as an advanced AI face verification and deepfake detection.

    Output Format:
    {{
        "verified": true/false,
        "similarity": "a value between 0.0 and 1.0",
        "confidence_score": "a value between 0.0 and 1.0",
        "explanation": "A brief justification summarizing your analysis."
    }}

    Input JSON:
    {faceDetectionModel}
    """

    try:
        client = OpenAI(
            api_key="api-key"
        )
        
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role":"user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_original}", "detail": "low"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_second}", "detail": "low"}
                        }
                    ]                
                }
            ],
            temperature=0.5
        )
        
        # parse response to json
        response_content = chat_completion.choices[0].message.content
        
        # remove markdown format from json response
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]

        # remove additional parts 
        response_content = response_content.strip()

        result = json.loads(response_content)

        # format response to model
        llm_result_model = models.LLMResultModel(
            verified=result.get("verified"),
            similarity=result.get("similarity"),
            confidence_score=result.get("confidence_score"),
            explanation=result.get("explanation"),
            test_valid=True
        )
        
        return llm_result_model
    except Exception as e:
        return models.LLMResultModel(
            verified=False, 
            explanation=f"An error occurred during verification: {str(e)}",
            similarity=0,
            confidence_score=0,
            test_valid=False
        )
        
class DetectorBackend(Enum):
    OPENCV = "1"
    RETINAFACE = "2"

@app.post("/modelValidation/")
async def validate_models(file1: UploadFile, file2: UploadFile, detectorBackend: DetectorBackend) -> list[models.DeepFaceVerificationModel]:
    # Read uploaded images into memory
    file1_content = await file1.read()
    file2_content = await file2.read()

    # Save images as temporary files for processing
    with open("file1_temp.jpg", "wb") as f:
        f.write(file1_content)
    with open("file2_temp.jpg", "wb") as f:
        f.write(file2_content)
    
    detectorBackendStr = ""
    if detectorBackend == DetectorBackend.OPENCV:
        detectorBackendStr = "opencv"
    elif detectorBackend == DetectorBackend.RETINAFACE:
        detectorBackendStr = "retinaface"
        
    deepface_models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        #"DeepFace", not tested because older version of tensorflow needed
        "DeepID", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        "GhostFaceNet"
    ]
    
    deepface_verification_list = []
    
    # Run DeepFace verifications to check which performs best
    for model in deepface_models:
        result = DeepFace.verify("file1_temp.jpg", "file2_temp.jpg", model_name=model, detector_backend=detectorBackendStr,)
        
        deepface_verification_model = models.DeepFaceVerificationModel(
            model=result["model"],
            deepface_verify= result["verified"],
            threshold=result["threshold"],
            cosine_distance= result["distance"],
            detector_backend=result["detector_backend"]            
        )
        
        deepface_verification_list.append(deepface_verification_model)

    return deepface_verification_list

#region landmarks

def extract_landmarks_from_face(image):
    # Extract facial landmarks from an image using Mediapipe's Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # Convert the BGR image to RGB before processing.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        results = face_mesh.process(image_rgb)
        
        FACEMESH_INDICES = {
            "silhouette": [
                10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
            ],

            "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
            "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
            "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

            "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
            "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
            "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
            "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
            "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
            "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
            "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],

            "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
            "rightEyebrowLower": [35, 124, 46, 53, 52, 65],

            "rightEyeIris": [473, 474, 475, 476, 477],

            "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
            "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
            "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
            "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
            "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
            "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
            "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],

            "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
            "leftEyebrowLower": [265, 353, 276, 283, 282, 295],

            "leftEyeIris": [468, 469, 470, 471, 472],

            "midwayBetweenEyes": [168],

            "noseTip": [1],
            "noseBottom": [2],
            "noseRightCorner": [98],
            "noseLeftCorner": [327],

            "rightCheek": [205],
            "leftCheek": [425]
        }

        landmarks_data = {}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for feature, indices in FACEMESH_INDICES.items():
                    landmarks_data[feature] = [
                    {
                        "x": face_landmarks.landmark[i].x, 
                        "y": face_landmarks.landmark[i].y
                    }
                    for i in indices
            ]
            
        return models.FaceLandmarks.from_dict(landmarks_data)
                
def get_landmarks(image1, image2):  
    # crop faces to 128x128 format
    face1 = crop_face(image1)
    face2 = crop_face(image2)
    
    if face1 is None or face2 is None:
        return 0
    
    try:
        landmarks1 = extract_landmarks_from_face(face1)
        landmarks2 = extract_landmarks_from_face(face2)
        
        return landmarks1, landmarks2
    except Exception as e:
        return None, None

#endregion

#region helper

def bytes_to_image(image_bytes):
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def crop_face(image):    
    try:
        detections = DeepFace.extract_faces(image)
    except Exception as e:
        print("Error in face recognition", e)
        return []
    
    for face in enumerate(detections):
        x, y, w, h = face[1]['facial_area']['x'], face[1]['facial_area']['y'], face[1]['facial_area']['w'], face[1]['facial_area']['h']

        # get face area and crop face
        x, y = max(0, x), max(0, y)
        w, h = min(image.shape[1] - x, w), min(image.shape[0] - y, h)
        face_crop = image[y:y+h, x:x+w]

        # resize face to 128 x 128 pixels
        face_resized = cv2.resize(face_crop, (128,128))
        return face_resized        

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

#endregion