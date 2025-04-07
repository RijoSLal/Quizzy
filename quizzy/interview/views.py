# from django.shortcuts import render
# from django.views import View
from django.http import JsonResponse,FileResponse,HttpRequest,HttpResponse
from rest_framework.views import APIView # type: ignore
from rest_framework.response import Response # type: ignore
from rest_framework.renderers import TemplateHTMLRenderer # type: ignore
from rest_framework import status # type: ignore
from django.http import StreamingHttpResponse
from . import no_stream_camera_capture, resume_management,retriever,speech,scrape,camera_capture
from django.shortcuts import redirect
from django.core.files.uploadedfile import UploadedFile
from dotenv import load_dotenv
import asyncio
import base64
import time
import whisper  # type: ignore
from io import BytesIO
import tempfile
from concurrent.futures import ThreadPoolExecutor
from reportlab.pdfgen import canvas
from textwrap import wrap
import logging
from django.contrib import messages
import numpy as np 
import cv2

logger = logging.getLogger("interview")
load_dotenv()

# Create your views here.


capture=no_stream_camera_capture.VideoCamera() #change to camera_capture for steaming webcamp data to server

rag_instance=retriever.RAG()
model = whisper.load_model("base")

class SessionMixin: 
    """Mixin to reset session-related data and updates."""

    def reset_session(self,request : HttpRequest) -> None:
        """Resets session validation status and removes unnecessary session data."""
        request.session["validation"] = False
        request.session["completed"] = True
        for key in ("history", "eval","time"):
            if key in request.session:
                del request.session[key]
        capture.reset_updates()
        rag_instance.score_reset()


class Home(APIView, SessionMixin):
    """
    API view for rendering the home page.

    - Uses 'TemplateHTMLRenderer' to return an HTML template.
    - Resets session validation on each request.
    - Removes chat history, evaluation, emotional probalilities.
    """ 
    renderer_classes=[TemplateHTMLRenderer]

    def get(self,request: HttpRequest) -> Response:
        logger.info("Loading home page.")
        self.reset_session(request)
        return Response(template_name="home.html")


class Myview(APIView, SessionMixin):
    """
    View class for handling resume validation.

    This class processes user-submitted resumes and job descriptions, calculates 
    their ATS (Applicant Tracking System) score using cosine similarity, and 
    determines whether the resume is eligible for further processing.

    """ 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resume_obj=resume_management.Resume()
        self.threshold=50
    """
        Initializes the Myview class.

        - resume_obj: Instance of the Resume class for resume processing.
        - threshold: Minimum ATS score required for validation.
    """
    renderer_classes=[TemplateHTMLRenderer]

    def get(self,request: HttpRequest) -> Response:
        
        """
        Handles GET requests to load the resume validation page.

        - Resets session validation status.
        - Returns the resume validation template.

        Returns:
            Response: Renders the "resume.html" template.
        """
        logger.info("Loading resume validation page.")
        self.reset_session(request)
        return Response(template_name="resume.html")
    
    
    def post(self,request: HttpRequest) -> HttpResponse:

        """
        Handles POST requests for resume validation.

        The function:
        - Extracts the uploaded resume file, job description, user-selected experience level, and countdown timer.
        - Ensures all required fields are provided.
        - Validates the file format (must be a PDF).
        - Computes the ATS score using "resume_management.final()".
        - Updates session data with results.
        - If the ATS score meets the threshold, the resume is inserted into a document database, and the user is redirected to the interview page.
        - Otherwise, the user is redirected back to the eligibility page.

        Request Data:
            - filename (UploadedFile): User-submitted resume file (PDF only).
            - description (str): Job description text.
            - choice (str): Experience level (beginner, intermediate, advance, expert).
            - time (str): Countdown timer value.

        Session Data:
            - user (dict): Extracted resume details.
            - countdown (int): Adjusted countdown timer (between 10 and 60 seconds).
            - position (list[int]): Experience level mapping.
            - ATS (int): Computed ATS score.
            - validation` (bool): Resume validation status.

        Returns:
            - Redirects to "interview" if the ATS score is sufficient.
            - Redirects to "eligibility" if any validation fails.

        Raises:
            - Exception: Catches unexpected errors and redirects the user with an error message.
        """
        try:
            logger.info("Processing resume validation.")
            file: UploadedFile | None = request.FILES.get("filename")
            description: str | None = request.POST.get("description")
            options: str | None = request.POST.get("choice")
            countdown: str | None = request.POST.get("time")

            missing_fields = [field for field, value in {
                "filename": file,
                "description": description,
                "choice": options,
                "time": countdown
            }.items() if not value]

            if missing_fields:
                logger.warning(f"Missing fields: {missing_fields}")
                messages.error(request, "All fields are required.")
                return redirect("eligibility")

            if file.content_type != "application/pdf" and not file.name.endswith(".pdf"):
                logger.warning(f"Invalid file type uploaded: {file.content_type}")
                messages.error(request, "Invalid file type. Please upload a PDF.")
                return redirect("eligibility")
            
            try:
                countdown = max(10, min(60, int(countdown)))
            except:
                logger.warning(f"Invalid countdown value: {countdown}, setting to default (10).")
                countdown = 10

            position:dict[str, list[int]] = {
                "beginner":[1,2], 
                "intermediate":[3,4],
                "advance":[4,5],
                "expert":[5,6] 
            }

            try:
               ats_score, resume, job_description, dictionary = self.resume_obj.final(file,description)
            except TypeError:
                logger.error("ResumeManagement failed to retrieve relevent info")
                redirect("eligibility")
            
            request.session.update({
                "user":dictionary,
                "countdown":countdown,
                "position":position.get(options,[1,2,3])
            })

            rag_instance.level = options 
            rag_instance.resume=resume
            summary=rag_instance.candidate_document_summarization()

            if ats_score >= self.threshold:
                logger.info(f"ATS validation passed with score: {ats_score}")
                
                request.session.update({
                    "ATS": ats_score,
                    "validation": True,
                    "summary":summary
                })
                rag_instance.document_insertion_chroma(resume,job_description)
                return redirect("interview")
            else:
                logger.warning(f"ATS validation failed with score: {ats_score}")
                messages.error(request, "ATS score is low")
                return redirect("eligibility")
            
        except Exception as e:
            logger.error(f"Unexpected error in resume validation: {str(e)}", exc_info=True)
            messages.error(request, "An unexpected error occurred. Please try again later.")
            return redirect("eligibility")




class Interview(APIView):

    """
    Handles the interview process by rendering the interview page 
    and processing user responses using RAG (Retrieval-Augmented Generation).
    """
    renderer_classes=[TemplateHTMLRenderer]

    def get(self,request: HttpRequest) -> Response | HttpResponse: 
        """
        Renders the interview page if the user has passed validation.

        - Redirects to the eligibility page if the session is not validated.
        - Initializes the interview conversation with a greeting.
        - Retrieves ATS (Applicant Tracking System) score from the session.

        Args:
            request (HttpRequest): The incoming GET request.

        Returns:
            Response | HttpResponse: The rendered interview page or a redirect response.
        """
        if not request.session.get("validation") or not request.session.get("completed",True):
            logger.warning("Validation missing in session redirecting to eligibility page")
            return redirect("eligibility")
        

        user: str = request.session["user"]["candidate"]
        chat_history: list[dict[str, str]] =request.session.setdefault(
            "history",[{"speaker":"ai","chat":f"Hi {user}, shall we begin?"}]
            )
        ats_score: int = request.session["ATS"]
        summary: str = request.session["summary"]
        return Response(
                {"conversation": chat_history,"ats":ats_score,"summary":summary},
                template_name="interview.html"
            )
    # ,"time":60

    def post(self,request:  HttpRequest) -> JsonResponse | Response:
        """
        Processes user responses from the interview form.

        - Handles audio file input for speech-to-text transcription.
        - Uses RAG to generate follow-up questions based on the user's response.
        - Evaluates the user's answer and stores the evaluation.
        - Converts the AI-generated response into speech and encodes it in Base64.
        - Stores the conversation history and evaluation in the session.

        Args:
            request (HttpRequest): The incoming POST request.

        Returns:
            JsonResponse | Response: A JSON response with updated conversation data 
        
        """
        audio_file: UploadedFile | None = request.FILES.get('audio')
        sound: str | None =request.POST.get("option")
        
        chat_history: list = request.session.get("history", [])
        evaluation: list=request.session.setdefault("eval",[])
        
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio.flush()  # ensure all data is written before processing
                result: dict[str,str] = model.transcribe(temp_audio.name,language="en")
                user_response: str = result["text"]
                logger.info("Trascription successful")
        else:
            logger.error("Audio not recived in the server")
            return Response({"conversation": chat_history}, template_name="interview.html")
        
        if  user_response:
            chat_history.append({"speaker":"human","chat": user_response})
            previous_question: str=chat_history[-2]["chat"]
        
           #---------------------------------------------------------------------------------------
            with ThreadPoolExecutor() as executor:
                future_questions = executor.submit(rag_instance.socratic_conversation, previous_question, user_response)
                future_response = executor.submit(rag_instance.evaluate_answer, future_questions.result(), user_response)
                
            questions:str=future_questions.result()
            response:str=future_response.result()
            # -------------------------------------------------------------------------------------
            
            evaluation.append(base64.b64encode(response.encode('utf-8')).decode('utf-8'))
            logger.info("Evaluation added to session")
          
            if questions:
               chat_history.append({"speaker":"ai","chat":questions})
            else:
                chat_history.append({"speaker":"ai","chat":"thank you"})
            request.session.modified = True
      
          
        audio_bytes: bytes =asyncio.run(speech.text_to_speech(str(questions),sound))
        audio_base64: str = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info("Audio encoding to Base64 completed successfully")
        #jasonResponse is used to avoid page reloading when form is submitted
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({"conversation": chat_history,"audio": audio_base64})

        return Response(
            {"conversation": chat_history}, template_name="interview.html"
            )
    


    
class Score(APIView):

    """
    Handles the interview scoring process, evaluates candidate performance, 
    and provides job recommendations based on the interview results.
    """
    renderer_classes=[TemplateHTMLRenderer]
    
    def get(self, request : HttpRequest) -> JsonResponse | Response:
        """
        Retrieves interview scores, evaluates candidate performance, 
        and provides job recommendations. 

        - Redirects to the eligibility page if validation is not present in the session.
        - Computes the final interview score based on the evaluation count.
        - Extracts job recommendations based on the candidate's applied job roles.
        - Processes vision system data to assess perception and posture.
        - Passes all calculated metrics to the score template.

        Returns:
            Response: Renders the score.html template with evaluation results, 
            perception analysis, job recommendations, and ATS score.
        """
        logger.info("Score view accessed.")
        if not request.session.get("validation"):
            logger.warning("Validation missing in session redirecting to eligibility page")
            return redirect("eligibility")
        
        percentage = int((rag_instance.score / rag_instance.count) * 10) if rag_instance.count else 0
        jobs=request.session["user"]["job"]
        position=request.session.get("position",[1,2,3,4])
        scrape_instance=scrape.Scrape(job=jobs,pos=position)
        job_details=scrape_instance.data_extraction(10)

        vision_system=capture.p

        perception=abs(round((vision_system[0]+vision_system[1])-vision_system[2],2))
        
        posture=round((100-vision_system[3]),2)

        ats_score=request.session["ATS"]

        context={"jobs":job_details,
                 "progress_prediction":vision_system,
                 "score":percentage,
                 "perception":perception,
                 "posture":posture,
                 "ats":ats_score}
        logger.info("Score computation successful returning response")
        return Response(context,template_name="score.html" )

    def post(self, request : HttpRequest) -> FileResponse:
        """
        Generates a PDF report containing interview evaluations and allows the user to download it.

        - Retrieves the evaluation history from the session.
        - Decodes and formats evaluation responses for structured output.
        - Wraps text to prevent overflow in the generated PDF.
        - Saves the PDF in memory and returns it as a downloadable file.

        Returns:
            FileResponse: A downloadable PDF file named "evaluation.pdf" containing the 
            candidate's interview evaluation.
        """
        logger.info("Generating evaluation report PDF.")
        buffer = BytesIO()
        p = canvas.Canvas(buffer)
        subheading_x=50
        text_x=70
        start_y=750
        line_spacing=40
        max_width=50
        for ind,item in enumerate(request.session.get("eval",[])):
            text=base64.b64decode(item).decode('utf-8')
            p.setFont("Courier-Bold", 14)  # Monospace font for subheadings
            p.drawString(subheading_x, start_y, str(ind+1) + ":")  # Subheading on one line
            # Wrap text to prevent overflow
            wrapped_text = wrap(text, width=max_width)  
            p.setFont("Courier", 12)  # Monospace font for text
            for line in wrapped_text:
                start_y -= 20  # Move down for each line of text
                p.drawString(text_x, start_y, line)

            start_y -= line_spacing 
       

        p.showPage()
        p.save()

        buffer.seek(0)
        logger.info("PDF generation complete.")
        return FileResponse(buffer, as_attachment=True, filename="evaluation.pdf")

class Cam(APIView):
    """  
    This view class streams webcam data to "interview.html".  
    It generates frames from the webcam and returns them as a streaming HTTP response.  
    """  
    def get(self,request: HttpRequest) -> StreamingHttpResponse | Response:
        """  
        Handles GET requests to stream webcam frames.  

        Returns:  
            - StreamingHttpResponse: If the frames are successfully generated and streamed.  
            - Response: If an error occurs while streaming the camera feed, returning a 500 error.  
        """  
        try:
            # logger.info("Attempting to stream webcam data.")
            return StreamingHttpResponse(camera_capture.generate_frames(capture), content_type="multipart/x-mixed-replace;boundary=frame")
        except Exception as e: # This is bad!
            # logger.error(f"Camera stream error: {e}")
            return Response(
                {"error":"Camera stream error"},status=500
                ) 
        
class No_Stream_Cam(APIView):
    """
    API endpoint for handling non-backend streaming image data.
    
    This view accepts a POST request containing a base64-encoded image. It decodes the image,
    processes it into a NumPy array, and passes it to the `capture.get_frame` method for further handling.
    
    Attributes:
        None
    
    Methods:
        post(request: HttpRequest) -> Response:
            Handles the incoming image data, decodes it, and forwards the frame for processing.
    
    Request Payload:
        - image (str): Base64-encoded image string.
    
    Response:
        - 200 OK: {'status': 'success'} if the image is processed successfully.
        - 500 Internal Server Error: {'status': 'error'} if an error occurs.
    """
    def post(self, request: HttpRequest) -> Response:
        
        image_data = request.data.get('image', '')
        try:
            if not image_data:
                return Response({'status': 'error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
            _, encoded_data = image_data.split('base64,', 1)
            decoded_image = base64.b64decode(encoded_data)
            
            np_arr = np.frombuffer(decoded_image, np.uint8)
            
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            capture.get_frame(frame) 
            
            return Response({'status': 'success'}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error occured in the non backend streaming image processing : {e}")
            return Response({'status': 'error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)   
        
class PredictionView(APIView):
    """  
    This view class fetches and streams the latest prediction data to "interview.html".  
    """  
    def get(self, request: HttpRequest) -> JsonResponse:
        """  
        Handles GET requests to retrieve the latest prediction result.  

        Returns:  
            - JsonResponse: A JSON object containing the latest prediction value.  
        """  
        # logger.info("Fetching latest prediction.")
        prediction = capture.get_latest_prediction()
        # logger.info(f"Latest prediction retrieved : {prediction}")
        return JsonResponse(
            {"prediction": prediction}
            )
    
class Check(APIView):
    """  
    This view class calculates the duration of an interview session  
    and provides information on the remaining time and progress.  
    """ 
    def get(self,request: HttpRequest) -> JsonResponse:
        """  
        Handles GET requests to compute the interview session duration.  

        Returns:  
            - JsonResponse: A JSON object containing:  
                - redirect: A boolean indicating if the interview has reached its time limit.  
                - time: The remaining time (in seconds) before the session ends.  
                - time_progress: The percentage of elapsed time relative to the total duration.  
        """  
        # logger.info("Checking interview duration.")
        interview_duration=int(request.session.get("countdown",10))*60
        if not request.session.get("time"):
            request.session["time"]=time.time()
      
        elapse=time.time()-request.session.get("time")
        remaining_time = max(0, interview_duration - elapse)
        progress = min(100, (elapse / interview_duration) * 100) 
        if elapse >= interview_duration: # time limit of interview
            logger.info("Interview time limit reached. Redirecting.")
            del request.session["time"]
            for key in ("history", "eval"):
                if key in request.session:
                    del request.session[key]
            request.session["completed"]=False
            return JsonResponse({"redirect":True})
        # logger.info(f"Time remaining: {remaining_time} seconds. Progress: {progress}%")
        return JsonResponse(
                {'redirect':False,"time":remaining_time,"time_progress":progress}
            )
    

