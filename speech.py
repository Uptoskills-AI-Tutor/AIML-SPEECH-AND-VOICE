import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import spacy
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scipy.io.wavfile import write

# Ensure required nltk resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load NLP models
nlp = spacy.load("en_core_web_sm")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
bert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load Whisper model
whisper_model = whisper.load_model("tiny")

class DialogueContext:
    def __init__(self):
        self.history = []

    def update_context(self, user_input, system_response):
        self.history.append({"user": user_input, "system": system_response})
        if len(self.history) > 5:
            self.history.pop(0)

    def get_context(self):
        return self.history

dialogue = DialogueContext()

def detect_intent(text):
    labels = ["question", "command", "statement"]
    return intent_classifier(text, candidate_labels=labels)

def extract_named_entities(text):
    doc = nlp(text)
    return {ent.text: ent.label_ for ent in doc.ents}

def get_best_sense(sentence, ambiguous_word):
    synsets = wn.synsets(ambiguous_word)
    if not synsets:
        return None

    doc = nlp(sentence)
    context_words = " ".join([token.text for token in doc if token.is_alpha and token.text.lower() != ambiguous_word.lower()])
    enhanced_sentence = f"{sentence} Context: {context_words}"
    sentence_embedding = bert_model.encode(enhanced_sentence, convert_to_tensor=True)

    best_sense = None
    highest_score = -1

    for synset in synsets:
        sense_representation = f"{synset.definition()}. Examples: {' '.join(synset.examples())}. Related words: {' '.join([lemma.name().replace('_', ' ') for lemma in synset.lemmas()])}"
        sense_embedding = bert_model.encode(sense_representation, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(sentence_embedding, sense_embedding).item()

        if similarity_score > highest_score:
            highest_score = similarity_score
            best_sense = synset

    return best_sense

def disambiguate_text(sentence):
    words = [token.text for token in nlp(sentence) if token.is_alpha]
    results = {}
    for word in words:
        best_sense = get_best_sense(sentence, word)
        if best_sense:
            results[word] = {
                "selected_sense": best_sense.name(),
                "definition": best_sense.definition()
            }
    return results

def process_text(text):
    print("\nProcessing Text for Context Understanding...\n")
    
    intent = detect_intent(text)
    entities = extract_named_entities(text)
    disambiguation = disambiguate_text(text)

    print(f"Intent Recognition: {intent}")
    print(f"Named Entities: {entities}")
    print(f"Disambiguation: {disambiguation}")

    return {
        "intent": intent,
        "entities": entities,
        "disambiguation": disambiguation
    }

def record_and_transcribe(duration=4, samplerate=16000):
    try:
        if not sd.query_devices():
            raise RuntimeError("No input audio device found. Please check your microphone.")

        print("\nSpeak now... ")  
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()

        audio = (audio / np.iinfo(np.int16).max).astype(np.float32)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_path = temp_audio.name
            write(temp_path, samplerate, audio)

        result = whisper_model.transcribe(temp_path, language="en", fp16=False, condition_on_previous_text=False)
        os.remove(temp_path)

        if result is None or "text" not in result:
            print("Whisper returned None. Skipping...")
            return

        transcription = result.get("text", "").strip()

        if not transcription:
            print("Transcription: [No speech detected]")
            return

        print(f"Transcription: {transcription}")
        context_data = process_text(transcription)
        print(f"Context Data: {context_data}")

    except Exception as e:
        print(f"Error in recording/transcribing: {e}")

if __name__ == "__main__":
    try:
        while True:
            record_and_transcribe()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
