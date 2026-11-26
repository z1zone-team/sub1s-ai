class BaseAiException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InvalidAudioFormatException(BaseAiException):
    def __init__(self, message: str = "지원하지 않는 오디오 파일 형식입니다."):
        super().__init__(message=message, status_code=400)


class SttProcessingException(BaseAiException):
    def __init__(self, message: str = "STT 음성 변환 중 오류가 발생했습니다."):
        super().__init__(message=message, status_code=500)


class EmbeddingModelUnavailableException(BaseAiException):
    def __init__(self, message: str = "취향 임베딩 모델을 불러오지 못했습니다."):
        super().__init__(message=message, status_code=503)


class IntentNotUnderstoodException(BaseAiException):
    """받아 적은 문장에서 출발역과 도착역을 알아내지 못했다. 역을 지어내지 않고 알린다"""

    def __init__(self, message: str = "출발역과 도착역을 알아듣지 못했어요. 역 이름을 말씀해 주세요."):
        super().__init__(message=message, status_code=422)
