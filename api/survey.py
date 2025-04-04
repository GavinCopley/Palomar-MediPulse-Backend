from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from survey import Survey, db

router = APIRouter()

@router.post("/survey")
def create_survey(survey_data: dict, db: Session = Depends(db)):
    survey = Survey(**survey_data)
    try:
        survey.create()
        return {"message": "Survey created successfully", "survey": survey.read()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/surveys")
def get_surveys(db: Session = Depends(db)):
    surveys = db.query(Survey).all()
    return {"surveys": [survey.read() for survey in surveys]}

@router.put("/survey/{survey_id}")
def update_survey(survey_id: int, survey_data: dict, db: Session = Depends(db)):
    survey = db.query(Survey).filter(Survey.id == survey_id).first()
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")
    updated_survey = survey.update(survey_data)
    if updated_survey:
        return {"message": "Survey updated successfully", "survey": updated_survey.read()}
    raise HTTPException(status_code=400, detail="Error updating survey")

@router.delete("/survey/{survey_id}")
def delete_survey(survey_id: int, db: Session = Depends(db)):
    survey = db.query(Survey).filter(Survey.id == survey_id).first()
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")
    
    db.delete(survey)
    db.commit()
    return {"message": "Survey deleted successfully"}
