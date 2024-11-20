from flask import Flask, request, render_template, send_file
import os
from pydub import AudioSegment
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

nltk.download('punkt')

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to convert multimedia files to WAV format
def convert_to_wav(input_file):
    converted_audio = os.path.join(UPLOAD_FOLDER, "converted_audio.wav")
    if input_file.lower().endswith((".mp3", ".wav")):
        sound = AudioSegment.from_file(input_file)
        sound.export(converted_audio, format="wav")
    elif input_file.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
        video = VideoFileClip(input_file)
        audio = video.audio
        audio.write_audiofile(os.path.join(UPLOAD_FOLDER, "extracted_audio.wav"))
        sound = AudioSegment.from_file(os.path.join(UPLOAD_FOLDER, "extracted_audio.wav"))
        sound.export(converted_audio, format="wav")
        os.remove(os.path.join(UPLOAD_FOLDER, "extracted_audio.wav"))
    else:
        raise ValueError("Unsupported file format. Please use MP3, WAV, MP4, MKV, AVI, or MOV.")
    return converted_audio

# Function to transcribe audio
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition; {e}"

# Function to summarize text
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    summary_text = " ".join(str(sentence) for sentence in summary)
    return summary_text

# Function to create a formatted PDF
def create_pdf(original_text, summarized_text, output_path="summary_output.pdf"):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin_left = 72
    margin_right = 108
    margin_top = 72
    margin_bottom = 72
    y_position = height - margin_top

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left, y_position, "Original Text")
    c.setFont("Helvetica", 10)
    y_position -= 20
    lines = original_text.split('\n')
    for line in lines:
        text_object = c.beginText(margin_left, y_position)
        text_object.textLines(line)
        c.drawText(text_object)
        y_position -= 12
        if y_position <= margin_bottom:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = height - margin_top

    y_position -= 20
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left, y_position, "Summarized Text")
    c.setFont("Helvetica", 10)
    y_position -= 30
    lines = summarized_text.split('\n')
    for line in lines:
        text_object = c.beginText(margin_left, y_position)
        text_object.textLines(line)
        c.drawText(text_object)
        y_position -= 12
        if y_position <= margin_bottom:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = height - margin_top

    c.save()

# Route for file upload and processing
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded. Please try again.", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            # Convert file to WAV
            wav_file = convert_to_wav(file_path)
            # Transcribe audio
            transcription = transcribe_audio(wav_file)
            # Summarize transcription
            summary = summarize_text(transcription, sentence_count=2)
            # Create PDF
            output_pdf_path = os.path.join(UPLOAD_FOLDER, "summary_output.pdf")
            create_pdf(transcription, summary, output_pdf_path)
            return send_file(output_pdf_path, as_attachment=True)
        except Exception as e:
            return str(e), 500
        finally:
            os.remove(file_path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
