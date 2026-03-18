from fastapi import APIRouter, HTTPException
from app.models.schemas import SignatureFromPathRequest, RoomSignature
from app.core.pipeline import SignaturePipeline

router = APIRouter()
pipeline = SignaturePipeline()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/signature/from-file", response_model=RoomSignature)
def signature_from_file(payload: SignatureFromPathRequest):
    try:
        return pipeline.build_from_path(
            image_path=payload.image_path,
            save_debug_image=payload.save_debug_image,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image file not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))