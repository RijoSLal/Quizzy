from django.shortcuts import render
from django.views import View
from django.http import JsonResponse,FileResponse,HttpRequest,HttpResponse
import asyncio
from rest_framework.views import APIView # type: ignore
from rest_framework.response import Response # type: ignore
from rest_framework import status # type: ignore
from . import no_stream_camera_capture, resume_management,retriever,speech,scrape
from django.shortcuts import redirect
from django.core.files.uploadedfile import UploadedFile
from dotenv import load_dotenv
import base64
import time
from io import BytesIO
from reportlab.pdfgen import canvas
from textwrap import wrap
import logging
from django.contrib import messages
import numpy as np 
import cv2

logger = logging.getLogger("interview")
load_dotenv()

# Create your views here.


capture = no_stream_camera_capture.VideoCamera()  # change to camera_capture for steaming webcamp data to server

stt_generator = speech.STTGenerator()
tts_generator = speech.SpeechGenerator()

class SessionMixin: 
    """Mixin to reset session-related data and updates."""

    def reset_session(self,request : HttpRequest) -> None:
        """Resets session validation status and removes unnecessary session data."""
        request.session["validation"] = False
        request.session["completed"] = True
        for key in ("history", "eval","time", "startup_audio_played", "suggested_questions", "questions_asked", "total_score", "eval_count"):
            if key in request.session:
                del request.session[key]
        capture.reset_updates(request.session)


class Home(View, SessionMixin):
    """
    API view for rendering the home page.

    - Uses 'TemplateHTMLRenderer' to return an HTML template.
    - Resets session validation on each request.
    - Removes chat history, evaluation, emotional probalilities.
    """ 

    def get(self,request: HttpRequest) -> HttpResponse:
        logger.info("Loading home page.")
        self.reset_session(request)
        return render(request, "home.html")


class Myview(View, SessionMixin):
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

    async def get(self,request: HttpRequest) -> HttpResponse:
        
        """
        Handles GET requests to load the resume validation page.

        - Resets session validation status.
        - Returns the resume validation template.

        Returns:
            Response: Renders the "resume.html" template.
        """
        logger.info("Loading resume validation page.")
        self.reset_session(request)
        return render(request, "resume.html")
    
    
    async def post(self,request: HttpRequest) -> HttpResponse:

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
               ats_score, resume, job_description, dictionary = await self.resume_obj.final(file,description)
            except TypeError:
                logger.error("ResumeManagement failed to retrieve relevent info")
                return redirect("eligibility")
            
            request.session.update({
                "user":dictionary,
                "countdown":countdown,
                "level": options,
                "position":position.get(options,[1,2,3]),
                "completed": True
            })
            # Reset the timer for the new interview
            if "time" in request.session:
                del request.session["time"]
            
            # Ensure the session has a key to use for ChromaDB collection uniqueness
            if not request.session.session_key:
                request.session.create()

            rag_instance = retriever.RAG(session_id=request.session.session_key, level=options)
            rag_instance.resume=resume

            if ats_score >= self.threshold:
                logger.info(f"ATS validation passed with score: {ats_score}")
                
                request.session.update({
                    "ATS": ats_score,
                    "validation": True,
                    "summary": dictionary.get("summary", "No summary available")
                })
                
                logger.info("Inserting documents into Chroma...")
                rag_instance.document_insertion_chroma(resume,job_description)
                
                logger.info("Triggering prepare_questions...")
                # Prepare structured questions at the beginning
                suggested_questions = await rag_instance.prepare_questions(resume, job_description)
                request.session["suggested_questions"] = suggested_questions
                request.session["questions_asked"] = 0
                request.session.modified = True
                
                logger.info(f"Resume validation process complete. suggested_pool_len={len(suggested_questions)}, redirecting to interview.")
                return redirect("interview")
            else:
                logger.warning(f"ATS validation failed with score: {ats_score}")
                messages.error(request, "ATS score is low")
                return redirect("eligibility")
            
        except Exception as e:
            logger.error(f"Unexpected error in resume validation: {str(e)}", exc_info=True)
            messages.error(request, "An unexpected error occurred. Please try again later.")
            return redirect("eligibility")




class TranscribeView(View):
    """
    Handles speech-to-text transcription as a separate step before LLM generation.
    """
    async def post(self, request: HttpRequest) -> JsonResponse:
        # Check if time is already up
        start_time = request.session.get("time")
        interview_duration = int(request.session.get("countdown", 10)) * 60
        if start_time and (time.time() - start_time >= interview_duration):
            return JsonResponse({"text": "TIMEOUT_SKIP", "redirect": True})

        audio_file: UploadedFile | None = request.FILES.get('audio')
        if not audio_file:
            return JsonResponse({"error": "No audio provided"}, status=400)
            
        result: dict[str,str] = await stt_generator.transcribe(audio_file, language="en")
        user_response: str = result.get("text", "")
        
        chat_history: list = request.session.get("history", [])
        if user_response:
            chat_history.append({"role": "user", "content": user_response})
            request.session["history"] = chat_history
            request.session.modified = True
            
        return JsonResponse({"text": user_response})

class Interview(View):

    """
    Handles the interview process by rendering the interview page 
    and processing user responses using RAG (Retrieval-Augmented Generation).
    """

    async def get(self,request: HttpRequest) -> HttpResponse: 
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
        domain: str = request.session.get("user", {}).get("job")
        level: str = request.session.get("level")
        suggested_questions = request.session.get("suggested_questions", [])
        
        logger.info(f"Interview GET: suggested_pool_len={len(suggested_questions)}")
        
        rag_instance = retriever.RAG(session_id=request.session.session_key, level=level)

        initial_history: list[dict[str, str]] = [
            {"role": "system", "content": rag_instance.get_system_prompt(is_final=False, domain=domain, level=level)},
            {"role": "assistant", "content": f"Hi {user}, are you ready for the interview?"}
        ]
        chat_history: list[dict[str, str]] = request.session.setdefault("history", initial_history)
        ats_score: int = request.session["ATS"]
        summary: str = request.session["summary"]
        
        # Filter out system prompt for display
        display_history = [msg for msg in chat_history if msg["role"] != "system"]

        startup_audio = ""
        if not request.session.get("startup_audio_played"):
            # Use voice from session if available, default to male
            session_voice = request.session.get("voice", "male")
            # Generate startup audio for the initial greeting
            startup_message = initial_history[1]["content"]
            audio_bytes: bytes = await tts_generator.text_to_speech(startup_message, voice=session_voice)
            startup_audio = base64.b64encode(audio_bytes).decode('utf-8')
            request.session["startup_audio_played"] = True

        # Initialize interview timer if not already set
        if "time" not in request.session:
            request.session["time"] = time.time()
            request.session.modified = True
        
        return render(
                request,
                "interview.html",
                {"conversation": display_history,"ats":ats_score,"summary":summary,"startup_audio": startup_audio}
            )
    # ,"time":60

    async def post(self,request:  HttpRequest) -> JsonResponse | HttpResponse:
        """
        Processes user responses from the interview form.

        - Uses RAG to generate follow-up questions based on the user's response.
        - Evaluates the user's answer and stores the evaluation.
        - Converts the AI-generated response into speech and encodes it in Base64.
        - Stores the conversation history and evaluation in the session.

        Args:
            request (HttpRequest): The incoming POST request.

        Returns:
            JsonResponse | Response: A JSON response with updated conversation data

        """
        # Check if time is already up before processing LLM
        start_time = request.session.get("time")
        interview_duration = int(request.session.get("countdown", 10)) * 60
        if start_time and (time.time() - start_time >= interview_duration):
             return JsonResponse({"redirect": True})

        sound: str | None =request.POST.get("option")
        if sound:
            request.session["voice"] = sound
            request.session.modified = True
            
        chat_history: list = request.session.get("history", [])

        if not chat_history or chat_history[-1]["role"] != "user":
            logger.error("No recent user message found in history")
            return JsonResponse({"error": "No user message found"}, status=400)

        #---------------------------------------------------------------------------------------
        suggested_questions = request.session.get("suggested_questions")
        questions_asked = request.session.get("questions_asked", 0)
        
        rag_instance = retriever.RAG(session_id=request.session.session_key)

        # Safety check: if suggested_questions is missing or empty but we haven't reached 15 questions, re-fill it.
        if (not suggested_questions) and questions_asked < 15:
            logger.warning(f"suggested_questions missing/empty in session at turn {questions_asked}. Re-loading from fallback.")
            suggested_questions = rag_instance.fallback_questions
            request.session["suggested_questions"] = suggested_questions

        logger.info(f"Interview Turn: questions_asked={questions_asked}, suggested_pool_len={len(suggested_questions) if suggested_questions else 0}")

        # Track progress
        request.session["questions_asked"] = questions_asked + 1

        # Completion is strictly determined by the server-side counter
        interview_complete = False
        candidate_name = request.session.get("user", {}).get("candidate")
        domain = request.session.get("user", {}).get("job")
        level = request.session.get("level")

        if questions_asked >= 14: # Turn 15 (0-indexed) is the final question
             logger.info("Final question reached. Setting completion flag.")
             interview_complete = True
             # Ensure the AI says its final goodbye
             questions = await rag_instance.socratic_conversation(chat_history, [], candidate_name=candidate_name, domain=domain, level=level)
        else:
            questions = await rag_instance.socratic_conversation(chat_history, suggested_questions, candidate_name=candidate_name, domain=domain, level=level)
            if suggested_questions:
                suggested_questions.pop(0)
                request.session["suggested_questions"] = suggested_questions
        
        logger.info(f"LLM Response: {questions[:100]}...")

        # -------------------------------------------------------------------------------------
        # EVALUATION DEFERRED TO SCORE PAGE
        # -------------------------------------------------------------------------------------

        if questions:
           chat_history.append({"role": "assistant", "content": questions})
        else:
            chat_history.append({"role": "assistant", "content": "thank you"})
        
        # Save state
        request.session["history"] = chat_history
        request.session.modified = True


        if not questions:
            questions = "I apologize, but I encountered an error. Could you please repeat that?"

        audio_bytes: bytes = await tts_generator.text_to_speech(questions, voice=sound or "male")
        audio_base64: str = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info("Audio encoding to Base64 completed successfully")

        # Filter out system prompt for display in AJAX response
        display_history = [msg for msg in chat_history if msg["role"] != "system"]

        #jasonResponse is used to avoid page reloading when form is submitted
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                "conversation": display_history,
                "audio": audio_base64,
                "redirect": interview_complete
            })
        return render(request, "interview.html", {"conversation": display_history})
    


    
class BatchEvaluationView(View):
    """
    Evaluates the entire interview history in one batch.
    """
    async def get(self, request: HttpRequest) -> JsonResponse:
        chat_history = request.session.get("history", [])
        if not chat_history:
            return JsonResponse({"status": "no_history"})

        evaluations = []
        total_score = 0
        count = 0
        
        rag_instance = retriever.RAG(session_id=request.session.session_key)

        user_message_count = 0
        # Iterate through history to find Q&A pairs
        # Only evaluate actual questions and answers, ignoring greetings and the final goodbye.
        for i in range(len(chat_history)):
            if chat_history[i]["role"] == "user":
                user_message_count += 1
                
                # Skip the first user message (usually "Yes" to "Are you ready?")
                if user_message_count == 1:
                    continue
                    
                user_answer = chat_history[i]["content"]
                # The question is the assistant message preceding this user answer
                question = ""
                for j in range(i-1, -1, -1):
                    if chat_history[j]["role"] == "assistant":
                        question = chat_history[j]["content"]
                        break

                if question:
                    res = await rag_instance.evaluate_answer(question, user_answer)
                    if res:
                        total_score += res["score"]
                        count += 1
                        # Encode for PDF generator compatibility
                        eval_str = f"Score: {res['score']}\nReason: {res['reason']}\nBetter Answer: {res['better']}\nFeedback: {res['communication_feedback']}"
                        evaluations.append(base64.b64encode(eval_str.encode('utf-8')).decode('utf-8'))

        request.session["eval"] = evaluations
        request.session["total_score"] = total_score
        request.session["eval_count"] = count
        request.session.modified = True

        if count == 0:
            return JsonResponse({"status": "no_history"})

        percentage = int((total_score / count) * 10)
        return JsonResponse({
            "status": "complete",
            "score": percentage,
            "eval_count": count
        })

class Score(View):

    """
    Handles the interview scoring process, evaluates candidate performance,
    and provides job recommendations based on the interview results.
    """
    def get(self, request : HttpRequest) -> HttpResponse:
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

        total_score = request.session.get("total_score", 0)
        count = request.session.get("eval_count", 0)
        percentage = int((total_score / count) * 10) if count else 0

        jobs=request.session["user"]["job"]
        position=request.session.get("position",[1,2,3,4])
        scrape_instance=scrape.Scrape(job=jobs,pos=position)
        job_details=scrape_instance.data_extraction(10)

        vision_system=capture.get_latest_prediction(request.session)

        perception=abs(round((vision_system[0]+vision_system[1])-vision_system[2],2))

        posture=round((100-vision_system[3]),2)

        ats_score=request.session["ATS"]

        context={"jobs":job_details,
                 "progress_prediction":vision_system,
                 "score":percentage,
                 "perception":perception,
                 "posture":posture,
                 "ats":ats_score,
                 "eval_count": count,
                 "eval_ready": "eval" in request.session}

        logger.info("Score computation successful returning response")
        return render(request, "score.html", context)

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

# class Cam(APIView):
#     """  
#     This view class streams webcam data to "interview.html".  
#     It generates frames from the webcam and returns them as a streaming HTTP response.  
#     """  
#     def get(self,request: HttpRequest) -> StreamingHttpResponse | Response:
#         """  
#         Handles GET requests to stream webcam frames.  

#         Returns:  
#             - StreamingHttpResponse: If the frames are successfully generated and streamed.  
#             - Response: If an error occurs while streaming the camera feed, returning a 500 error.  
#         """  
#         try:
#             # logger.info("Attempting to stream webcam data.")
#             return StreamingHttpResponse(camera_capture.generate_frames(capture), content_type="multipart/x-mixed-replace;boundary=frame")
#         except Exception as e: # This is bad!
#             # logger.error(f"Camera stream error: {e}")
#             return Response(
#                 {"error":"Camera stream error"},status=500
#                 ) 
        
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

            # High-precision rate limiting: ignore frames if sent too frequently (min 0.45s interval)
            current_time = time.perf_counter()
            last_frame_time = request.session.get('last_frame_time', 0)
            
            if current_time - last_frame_time < 0.45:
                return Response({'status': 'ignored'}, status=status.HTTP_200_OK)

            request.session['last_frame_time'] = current_time
        
            _, encoded_data = image_data.split('base64,', 1)
            decoded_image = base64.b64decode(encoded_data)
            
            np_arr = np.frombuffer(decoded_image, np.uint8)
            
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            capture.get_frame(frame, request.session)

            return Response({'status': 'success'}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error occured in the non backend streaming image processing : {e}")
            return Response({'status': 'error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)   
        
class PredictionView(View):
    """
    This view class fetches and streams the latest prediction data to "interview.html".
    """
    async def get(self, request: HttpRequest) -> JsonResponse:
        """
        Handles GET requests to retrieve the latest prediction result.

        Returns:
            - JsonResponse: A JSON object containing the latest prediction value.
        """
        # logger.info("Fetching latest prediction.")
        prediction = capture.get_latest_prediction(request.session)
        # logger.info(f"Latest prediction retrieved : {prediction}")
        return JsonResponse(
            {"prediction": prediction}
            )

class Check(View):
    """
    This view class calculates the duration of an interview session
    and provides information on the remaining time and progress.
    """
    async def get(self,request: HttpRequest) -> JsonResponse:
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

        # Ensure start time is set and persisted exactly once per interview
        if "time" not in request.session:
            request.session["time"] = time.time()
            request.session.modified = True

        start_time = request.session["time"]

        elapse=time.time()-start_time
        remaining_time = max(0, interview_duration - elapse)
        progress = min(100, (elapse / interview_duration) * 100) 
        if elapse >= interview_duration: # time limit of interview
            logger.info("Interview time limit reached. Preparing final goodbye.")
            request.session["completed"] = False # Prevents re-entering the interview
            request.session.modified = True
            
            # Load RAG instance to get access to fallback goodbyes
            rag_instance = retriever.RAG(session_id=request.session.session_key)
            import random
            final_msg = random.choice(rag_instance.fallback_goodbyes)
            
            # Prepend a notice that time is up
            final_msg = f"The interview time is up. {final_msg}"

            # Generate audio bytes asynchronously
            session_voice = request.session.get("voice", "male")
            audio_bytes = await tts_generator.text_to_speech(final_msg, voice=session_voice)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            return JsonResponse({
                "redirect": True,
                "message": final_msg,
                "audio": audio_base64
            })
        # logger.info(f"Time remaining: {remaining_time} seconds. Progress: {progress}%")
        return JsonResponse(
                {'redirect':False,"time":remaining_time,"time_progress":progress}
            )

