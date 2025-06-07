from pydantic import BaseModel
from typing import Optional

class LandmarkPoint(BaseModel):
    x: float
    y: float

class FaceLandmarks(BaseModel):
    image_shape: str
    silhouette: list[LandmarkPoint]
    lipsUpperOuter: list[LandmarkPoint]
    lipsLowerOuter: list[LandmarkPoint]
    lipsUpperInner: list[LandmarkPoint]
    lipsLowerInner: list[LandmarkPoint]
    rightEyeUpper0: list[LandmarkPoint]
    rightEyeLower0: list[LandmarkPoint]
    rightEyeUpper1: list[LandmarkPoint]
    rightEyeLower1: list[LandmarkPoint]
    rightEyeUpper2: list[LandmarkPoint]
    rightEyeLower2: list[LandmarkPoint]
    rightEyeLower3: list[LandmarkPoint]
    rightEyebrowUpper: list[LandmarkPoint]
    rightEyebrowLower: list[LandmarkPoint]
    rightEyeIris: list[LandmarkPoint]
    leftEyeUpper0: list[LandmarkPoint]
    leftEyeLower0: list[LandmarkPoint]
    leftEyeUpper1: list[LandmarkPoint]
    leftEyeLower1: list[LandmarkPoint]
    leftEyeUpper2: list[LandmarkPoint]
    leftEyeLower2: list[LandmarkPoint]
    leftEyeLower3: list[LandmarkPoint]
    leftEyebrowUpper: list[LandmarkPoint]
    leftEyebrowLower: list[LandmarkPoint]
    leftEyeIris: list[LandmarkPoint]
    midwayBetweenEyes: list[LandmarkPoint]
    noseTip: list[LandmarkPoint]
    noseBottom: list[LandmarkPoint]
    noseRightCorner: list[LandmarkPoint]
    noseLeftCorner: list[LandmarkPoint]
    rightCheek: list[LandmarkPoint]
    leftCheek: list[LandmarkPoint]
    
    @classmethod
    def from_dict(cls, data):
        #convert dict to FaceLandmarks object
        return cls(
            image_shape="128x128",
            **{key: [LandmarkPoint(**point) for point in value] for key, value in data.items()}
        )
    
class DeepFaceVerificationModel(BaseModel):
    model: str
    deepface_verify: bool
    threshold:float
    cosine_distance: float
    detector_backend: str

class FaceDetectionModel(BaseModel):
    deepface_verification_list: list[DeepFaceVerificationModel]
    face1_face_landmarks: Optional[FaceLandmarks] = None
    face2_face_landmarks: Optional[FaceLandmarks] = None
    
class LLMResultModel(BaseModel):
    verified: bool
    similarity: float
    confidence_score: float
    explanation: str
    test_valid: bool

class DeepFaceValidationTestModel(BaseModel):
    deepface_verification_test: list[DeepFaceVerificationModel]