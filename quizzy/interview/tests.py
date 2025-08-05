from django.test import TestCase,Client
from django.urls import reverse
from . import resume_management,vectordb,speech,scrape,retriever,camera_capture
from django.core.files.uploadedfile import SimpleUploadedFile
import io
import asyncio
from reportlab.pdfgen import canvas
import cv2
# Create your tests here.
client=Client()

class VectorDB(TestCase):
    """
    TestCase for file vectordb.py which deals with chromadb operations
    """
    def setUp(self):
        self.chroma=vectordb.ChromaDB()
        self.extracted_text="This is a test document. It contains multiple sentences.\nNew paragraph starts here"

    
    def test_insertion_docs_delete_inserted_docs(self):
        """
        Test the insertion and deletion of documents  
        """
        self.chroma.insert_into_chroma(self.extracted_text)
        before_delete = self.chroma.vectorstore.get()
        self.assertGreater(len(before_delete["ids"]), 0)
        self.chroma.delete_inserted_docs()
        after_delete = self.chroma.vectorstore.get()
        self.assertEqual(after_delete["ids"], []) 

    def test_get_all_document(self):
        """
        Test the "get_all_documents()" method retrieves all document in the vectordb
        """
        documents = self.chroma.get_all_documents()
        self.assertIsInstance(documents,list)


    def test_get_random_document(self):
        """
        Test the random document from the get_random_document() in vectordb is not repeated
        """
        self.chroma.insert_into_chroma(self.extracted_text)
        all_documents = self.chroma.get_all_documents()
        num_documents = len(all_documents)

        retrieved_docs = set()
        for _ in range(num_documents):
            doc_content = self.chroma.get_random_document()
            self.assertIsNotNone(doc_content)
            retrieved_docs.add(doc_content)

        self.assertEqual(len(retrieved_docs), num_documents)
        self.assertIsNone(self.chroma.get_random_document())
    

class ResumeManagement(TestCase):
    """
    TestCase for file resume_management.py which deals with evaluation of resume and validation
    """
     
    def setUp(self):
        self.resume=resume_management.Resume()
        self.job_description="Looking for a Python developer with experience in Django and machine learning."
        self.resume_details="Jhon Wick is a Software developer working in ML and Django"
        

    def test_resume_reader(self): 
        """
        Test resume reader works
        """
        fake_pdf = self.create_fake_pdf()  
        readed_file=self.resume.resume_reader(fake_pdf)
        self.assertIsInstance(readed_file,(str,type(None)))
    
    def test_domain_name_extraction(self):
        "Test domain and name extraction with ollama model works"
        candidate_doc=" ".join([self.resume_details,self.job_description])
        extracted_domain_name=self.resume.domain_name_extraction(candidate_doc)
        self.assertIsInstance(extracted_domain_name,dict)
        self.assertIn("candidate",extracted_domain_name)
        self.assertIn("job",extracted_domain_name)

    def test_ats_score(self):
        """
        Test ATS processing works
        """
        score=self.resume.ats_score_checker(self.resume_details,self.job_description)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def create_fake_pdf(self):
        """
        Create fake PDF for testing
        """
        fake_pdf = io.BytesIO()
        c = canvas.Canvas(fake_pdf)
        c.drawString(100, 750, "John Doe - 5+ years experience in Django & Machine Learning") 
        c.save()
        fake_pdf.seek(0)

        return fake_pdf


    def test_final_method(self):
        fake_pdf = self.create_fake_pdf()
        result = self.resume.final(fake_pdf, self.job_description)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4) 
        score, resume_text, description, extracted_details = result

        self.assertIsInstance(score, (int, float)) 
        self.assertIsInstance(resume_text, str) 
        self.assertEqual(description, self.job_description) 
        self.assertIsInstance(extracted_details, dict) 


class TTS(TestCase):
    """
    Testcase for speech.py which converts text to speech and returns as bytes
    """
    def test_tts(self):
        """
        Sample text test
        """
        tts_text="Hello world"
        tts_output=asyncio.run(speech.text_to_speech(tts_text))
        self.assertIsInstance(tts_output,bytes)

class Scraper(TestCase):
    """
    Testcase for scrape.py which scrapes job search data from linkdin
    """
    def setUp(self):
        self.scraper = scrape.Scrape("AI",[1,2,3])
        self.n=5

    def test_list_items(self):
        """
        Test scraping works by checking returning list
        """
        job_items = self.scraper.list_items()
        self.assertIsInstance(job_items, list) 

        if job_items:  
 
            self.assertGreater(len(job_items), 0, "Job listings should not be empty.")
    
    def test_data_extraction(self):
        """
        Test scraping returns in desired format 
        """
        extracted_jobs = self.scraper.data_extraction(self.n)  
        self.assertIsInstance(extracted_jobs, list)  
        for job in extracted_jobs:
            self.assertIsInstance(job, dict) 
            self.assertIn("job", job)  
            self.assertIn("company", job)
            self.assertIn("link", job)
            self.assertIn("location", job)

# class VideoCameraTestCase(TestCase):
#     """
#     Testcase for camera_capture.py which handles add vision based processing and emotion prediction
#     """
#     def setUp(self):
#         self.cam = camera_capture.VideoCamera()
#     def test_singleton(self):

#         """test that VideoCamera follows a singleton pattern."""

#         cam1 = camera_capture.VideoCamera()
#         cam2 = camera_capture.VideoCamera()
#         self.assertIs(cam1, cam2)

#     def test_camera_initialization(self):

#         """test the camera initializes correctly."""

#         self.assertIsInstance(self.cam.video, cv2.VideoCapture)
#         self.assertTrue(self.cam.video.isOpened())

#     def test_detect_head_down(self):
       
#         """
#         Test the posture prediction works correctly
#         """

#         nose = (100, 120)
#         left_eye = (90, 100)
#         right_eye = (110, 100)
#         self.assertTrue(self.cam.detect_head_down(nose, left_eye, right_eye))

     
#         nose = (100, 80)
#         self.assertFalse(self.cam.detect_head_down(nose, left_eye, right_eye))

#     def test_upgrade(self):
#         """
#         Test scores are updating correctly
#         """
#         self.cam.reset_updates()
#         self.cam.upgrade(emo=1, pos=True)
#         self.cam.upgrade(emo=2, pos=False)
#         self.cam.upgrade(emo=1, pos=True)

#         self.assertEqual(self.cam.counts, [0, 2, 1])  
#         self.assertEqual(self.cam.total, 3)
#         self.assertEqual(self.cam.pos_count, 2)
#         self.assertTrue(0 <= sum(self.cam.p) <=200) 

#     def test_get_latest_prediction(self):

#         """
#         Test score retrieval works correctly
#         """
     
#         cam = camera_capture.VideoCamera()
#         cam.upgrade(emo=0, pos=False)
#         cam.upgrade(emo=1, pos=True)

#         probabilities = cam.get_latest_prediction()
#         self.assertEqual(len(probabilities), 4) 

#     def test_video_streaming(self):
#         """
#         Test video streaming works correctly
#         """
#         frame = self.cam.get_frame()
#         self.assertIsInstance(frame, bytes) 
#         self.assertTrue(len(frame) > 0)  

#     @classmethod
#     def tearDownClass(cls):
#         """just to ensure camera releases proper after testing"""
#         camera_capture.VideoCamera().release()



# class RAG_Test(TestCase):
#     """Tests for the RAG (Retrieval-Augmented Generation) component"""

#     def setUp(self):
#         self.retriever_instance=retriever.RAG()
#         self.question = "What is overfitting in ML?"
#         self.response = "Overfitting is when a model memorizes training data, leading to poor generalization."
        
#     def Generate_interview_question_test(self):
#         """Test question generation produces expected output"""

#         document = "Regularization techniques like L1 and L2 help prevent overfitting by penalizing large weights in machine learning models."
        
#         question_gen = self.retriever_instance.generate_interview_questions(
#                         document, 
#                         self.question,
#                         self.response,
#         )
#         self.assertIsInstance(question_gen,(str,type(None)))
        
    # def Evaluate_answer_test(self):
    #     """Test answer evaluation produces expected output"""
    #     score = self.retriever_instance.evaluate_answer(
    #             self.question,
    #             self.response,
    #     )
    #     self.assertIsInstance(score,(str,type(None)))
    #     self.assertTrue(0 <= self.retriever_instance.score <= 10)
    #     self.assertGreater(self.retriever_instance.count)
        
    # def Socratic_conversation_test(self):
    #     """Test proper retrieval of socratic_conversation(previous_question, user_response)"""
    #     previous_question=self.question
    #     user_response=self.response
    #     follow_up_question = self.retriever_instance.socratic_conversation(previous_question, user_response)
    #     self.assertIsInstance(follow_up_question, (str, type(None)))

    # def Summary_test(self):
    #     """Test summarization produces expected output"""
    #     self.retriever_instance.resume="""
    #                                     Money plays a crucial role in shaping economies, societies, and individual lives.
    #                                     It is a universally accepted medium of exchange that facilitates trade and commerce, 
    #                                     allowing people to acquire goods and services without relying on the inefficiencies of bartering. 
    #                                     Throughout history, different forms of money have evolved, from metal coins and paper currency to digital transactions and cryptocurrencies. 
    #                                     Beyond its practical function, money also serves as a store of value,
    #                                     enabling individuals to save and plan for the future.
    #                                     Financial stability often brings security, freedom, and opportunities,
    #                                     allowing people to invest in education, healthcare, businesses, and personal aspirations.
    #                                     However, money is not just a tool for survival—it also influences human behavior, social structures, and even personal relationships.     
    #                                     The way people earn, spend, and manage money can reflect their values, priorities, and financial literacy. 
    #                                     While wealth can provide comfort and access to better opportunities, an excessive obsession with accumulating money can lead to stress, greed, and imbalance.
    #                                     Ultimately, money is a powerful yet neutral force—it can be used to create a better life, support loved ones,
    #                                     and contribute to society, or it can become a source of anxiety and division. Learning to manage money wisely,
    #                                     understanding its limitations, and maintaining a healthy perspective on wealth are essential for achieving financial well-being and a fulfilling life.
    #                                     """
    #     summary=self.retriever_instance.candidate_document_summarization()
    #     self.assertIsInstance(summary,str)

class HomeTest(TestCase):
    """Tests for the home page view"""

    def setUp(self):
        self.client=Client()
    def test_get(self):
        """Test if the home page returns a 200 status"""
        response=self.client.get(reverse("home"))
        self.assertEqual(response.status_code,200)
        self.assertTemplateUsed(response,"home.html")
        self.assertFalse(self.client.session.get("validation", True))

class MyViewTest(TestCase):
    """Test for eligibility page view"""
    def setUp(self):
        self.client=Client()
        self.url = reverse("eligibility")
        self.job_description="Looking for a Python developer with experience in Django and machine learning."
        
        
    def test_get(self):
        """Test if the eligibility page returns a 200 status"""
        response=self.client.get(reverse("eligibility"))
        self.assertEqual(response.status_code,200)
        self.assertTemplateUsed(response,"resume.html")
        self.assertFalse(self.client.session.get("validation", True))

    def create_fake_pdf(self):
        """Create fake PDF for test uses"""
        fake_pdf = io.BytesIO()
        c = canvas.Canvas(fake_pdf)
        c.drawString(100, 750, "John Doe - 5+ years experience in Django & Machine Learning") 
        c.save()
        fake_pdf.seek(0)

        return fake_pdf

    def test_post_request(self):
        """Test if the eligibility page returns a 200,300 status after POST request"""
        valid_resume=self.create_fake_pdf()
        valid_data = {
            "filename": valid_resume,
            "description": self.job_description,
            "choice": "intermediate",
            "time": "30"
        }
        response = self.client.post(self.url, valid_data, format='multipart')
        self.assertIn(response.status_code, [302, 200])



class InterviewTest(TestCase):
    """Tests for the interview session handling and other processes"""
    def setUp(self):
        self.client=Client()
        self.url = reverse("interview")
        self.cam_url=reverse("live")
        self.check_url= reverse("time")
        self.prediction_url=reverse("prediction")
        self.audio_file = SimpleUploadedFile("test.wav", b"dummy audio content", content_type="audio/wav")
        self.valid_session_data = {
            "validation": True,
            "user": {"candidate": "John Doe"},
            "history": [{"speaker": "ai", "chat": "Hi John Doe, shall we begin?"}],
            "ATS": 85,
            "summary":"N/A"
        }
        session = self.client.session
        session.update(self.valid_session_data)
        session.save()
    
    def test_get_request(self):
        """Test if the interview page returns a 200 status and renders valid html page"""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "interview.html")
        self.assertIsInstance(response.context["conversation"][0]["chat"], str)
        self.assertEqual(response.context["ats"], 85)

    def test_post_request(self):
        pass
        # data = {"audio": self.audio_file, "option": "default"}
        # response = self.client.post(self.url, data, format="multipart")

        # self.assertEqual(response.status_code, 200)
        # self.assertIn("conversation", response.json())
        # self.assertIn("audio", response.json())  

    def test_camera_stream(self):
        """Test if the camera endpoint returns data and in valid format"""
        response = self.client.get(self.cam_url)

        # Expect streaming response (status 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "multipart/x-mixed-replace;boundary=frame")

    def test_prediction_view(self):
        """
        Test if the score endpoint returns score data and in valid format
        """
        response = self.client.get(self.prediction_url)

        # Expect JSON response with a 'prediction' key
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    def test_check_view_with_time_remaining(self):
        """
        Test if the time endpoint returns time related data and in valid format
        """
        response = self.client.get(self.check_url)

        self.assertEqual(response.status_code, 200)
        self.assertIn("redirect", response.json())
        self.assertIn("time", response.json())
        self.assertIn("time_progress", response.json())

        self.assertIsInstance(response.json()["redirect"],bool)
    
class ScoreTest(TestCase):
    """Tests for score page and calculations"""
    def setUp(self):
        self.client=Client()
        self.valid_session_data = {
            "validation": True,
            "user": {"candidate": "John Doe", "job": "Software Engineer"},
            "position": [1, 2, 3, 4],
            "ATS": 85
        }
        session = self.client.session
        session.update(self.valid_session_data)
        session.save()

        
    def get_test(self):
        """
        Test if the score page returns 200 status and in valid format and renders correct html file
        """

        response=self.client.get(reverse("score"))
        self.assertEqual(response.status_code,200)
        self.assertTemplateUsed(response,"score.html")
        context=response.context
        self.assertIn("score", context)
        self.assertIn("ats", context)
        self.assertIn("perception", context)
        self.assertIn("posture", context)
        self.assertIn("jobs", context)
        self.assertEqual(context["ats"], 85)


    def post_test(self):
        """
        Test PDF downloading works by checking status code and return type
        """
        response = self.client.get(reverse("score"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("Content-Disposition", response)

