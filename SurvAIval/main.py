<<<<<<< HEAD
import subprocess
import traceback
import queue
import numpy as np
import logging
from typing import Optional, Tuple  # Corrected Tuple import
import threading
import time
import os  # Added for file path handling
import soundfile as sf  # Added for reading audio files

try:
    import sounddevice as sd
    import whisper
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Removed pyttsx3 import
    from TTS.api import TTS  # Added Coqui TTS import
except ImportError as e:
    print(f"Error importing required library: {e}")
    print(
        "Please ensure sounddevice, openai-whisper, torch, transformers, TTS (coqui-tts), and soundfile are installed."  # Updated requirements
    )
    print("You might also need FFmpeg installed on your system.")
    exit(1)

# --- Configuration ---
MODEL_NAME: str = "unsloth/DeepScaleR-1.5B-Preview"
WHISPER_MODEL_SIZE: str = (
    "medium"  # Options: tiny, base, small, medium, large (trade-off speed/accuracy)
=======
import queue
import numpy as np
import logging
from typing import Optional, Tuple
import threading
import time
import os
import soundfile as sf
import av  # used by faster-whisper
import traceback

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from TTS.api import TTS
except ImportError as e:
    print(f"Error importing required library: {e}")
    print(
        "Please ensure sounddevice, faster-whisper, torch, transformers, TTS (coqui-tts), and soundfile are installed."
    )
    print(
        "You might also need FFmpeg installed (or ensure the 'av' library provides it) and espeak-ng (for some TTS models)."  # Added espeak-ng note here
    )
    exit(1)

# --- Configuration ---
MODEL_NAME: str = "unsloth/Qwen2.5-3B-Instruct"
WHISPER_MODEL_SIZE: str = (
    "small"  # Options: tiny, base, small, medium, large (trade-off speed/accuracy)
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
)
WHISPER_LANGUAGE: str = (
    "en"  # Set to None for auto-detection, or specify a language code (e.g., 'fr', 'es')
)
SAMPLE_RATE: int = 16000
<<<<<<< HEAD
RECORD_SECONDS: int = 30  # Maximum recording duration
MAX_NEW_TOKENS: int = 512  # Reduced token limit for concise answers
=======
RECORD_SECONDS: int = 60  # Maximum recording duration
MAX_NEW_TOKENS: int = 1024  # Reduced token limit for concise answers
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
DEVICE_MAP: str = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Simplified device check
TORCH_DTYPE = (
    torch.float16 if DEVICE_MAP == "cuda" else torch.float32
)  # Use float32 on CPU

SYSTEM_PROMPT: str = (
    """You are an AI designed to assist with survival in emergency situations. Your expertise includes first aid, finding shelter, securing food and water, and other critical survival skills. Your responses should be clear, concise, and actionable, focused on providing practical advice for individuals in need."""
)

# --- Coqui TTS Configuration ---
# Choose a Coqui TTS model. Find more models at https://github.com/coqui-ai/TTS/blob/dev/TTS/server/model_manager.py
# Example models:
# "tts_models/en/ljspeech/tacotron2-DDC" (Female, US English, good quality)
# "tts_models/en/vctk/vits" (Multi-speaker, requires speaker selection)
# "tts_models/en/ek1/tacotron2" (Male, US English)
<<<<<<< HEAD
COQUI_TTS_MODEL_NAME: str = "tts_models/en/ljspeech/tacotron2-DDC"
# Optional: Specify a speaker if the model is multi-speaker
COQUI_TTS_SPEAKER: Optional[str] = None  # e.g., "p225" for vctk/vits
# Optional: Specify a language if the model is multi-lingual
=======
COQUI_TTS_MODEL_NAME: str = "tts_models/en/vctk/vits"
COQUI_TTS_SPEAKER: Optional[str] = "p225"  # e.g., "p225" for vctk/vits
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
COQUI_TTS_LANGUAGE: Optional[str] = "en" if "multi" in COQUI_TTS_MODEL_NAME else None

TTS_OUTPUT_FILENAME: str = "tts_output.wav"  # Temporary file for TTS audio

<<<<<<< HEAD
# Removed pyttsx3 specific config (TTS_VOICE, TTS_RATE)

MAX_HISTORY_LENGTH: int = 5  # Number of recent turns to keep in memory
# --- End Configuration ---

# --- Logging Setup ---
=======
MAX_HISTORY_LENGTH: int = 5
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

<<<<<<< HEAD
# --- Initialization ---
print("Initializing components...")


# Check for FFmpeg (required by Whisper)
def check_ffmpeg():
    """Checks if FFmpeg is accessible in the system path."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logging.info("FFmpeg found.")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.error("-" * 50)
        logging.error("ERROR: FFmpeg not found or not executable.")
        logging.error("FFmpeg is required by the Whisper model for audio processing.")
        logging.error("Please install FFmpeg for your operating system:")
        logging.error("  - Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg")
        logging.error("  - MacOS (Homebrew): brew install ffmpeg")
        logging.error(
            "  - Windows: Download from https://ffmpeg.org/download.html and add to PATH"
        )
        logging.error("-" * 50)
        return False


# Load Whisper Model
def load_whisper_model(model_size: str) -> Optional[whisper.Whisper]:
    """Loads the specified OpenAI Whisper model."""
    try:
        logging.info(f"Loading Whisper model ({model_size})...")
        # Load model to the determined device
        model = whisper.load_model(model_size, device=DEVICE_MAP)
        logging.info(f"Whisper model loaded successfully to {DEVICE_MAP}.")
        return model
    except Exception as e:
        logging.error(f"Error loading Whisper model: {e}")
=======

def load_whisper_model(
    model_size: str,
) -> Optional[WhisperModel]:  # Note: Return type changes
    """Loads the specified Faster Whisper model."""
    try:
        logging.info(f"Loading Faster Whisper model ({model_size})...")
        # Choose compute type based on device (float16 for CUDA, int8 or float32 for CPU)
        compute_type = (
            "float16" if DEVICE_MAP == "cuda" else "int8"
        )  # int8 is often faster on CPU
        # Load model to the determined device
        model = WhisperModel(model_size, device=DEVICE_MAP, compute_type=compute_type)
        logging.info(
            f"Faster Whisper model loaded successfully to {DEVICE_MAP} with compute_type {compute_type}."
        )
        return model
    except Exception as e:
        logging.error(f"Error loading Faster Whisper model: {e}")
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
        logging.error(traceback.format_exc())
        return None


<<<<<<< HEAD
=======
# --- (Keep load_language_model as is) ---
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
def load_language_model(
    model_name: str,
) -> Tuple[
    Optional[AutoModelForCausalLM], Optional[AutoTokenizer]
]:  # Corrected type hint
    """Loads the Hugging Face language model and tokenizer, ensuring chat template is set."""
    try:
        logging.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logging.info("Tokenizer loaded.")

        # --- (Keep the tokenizer chat_template and pad_token logic as is) ---
        if tokenizer.chat_template is None:
            logging.warning("Tokenizer missing chat_template. Applying a default.")
            default_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )
            tokenizer.chat_template = default_template
            logging.info("Default chat template applied.")

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                logging.info("Setting pad_token to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                logging.warning(
                    "EOS token missing. Adding a default pad token '<|pad|>'"
                )
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                # Also set eos_token if it's missing, crucial for generation
                if tokenizer.eos_token is None:
                    logging.warning("EOS token also missing. Adding '<|endoftext|>'.")
                    tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        # --- (End of tokenizer logic) ---

        logging.info(
            f"Loading language model {model_name} to {DEVICE_MAP} (this may take time)..."
        )

        # Prepare arguments for loading the model
        model_kwargs = {
            "torch_dtype": TORCH_DTYPE,
            "trust_remote_code": True,
        }

<<<<<<< HEAD
        # *** MODIFICATION START ***
=======
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
        # Only pass device_map and low_cpu_mem_usage if targeting CUDA
        if DEVICE_MAP == "cuda":
            model_kwargs["device_map"] = DEVICE_MAP
            # low_cpu_mem_usage is primarily beneficial when using device_map with CUDA
            model_kwargs["low_cpu_mem_usage"] = True
        # If DEVICE_MAP is 'cpu', we don't pass device_map, letting transformers default to CPU.
<<<<<<< HEAD
        # *** MODIFICATION END ***
=======
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718

        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs  # Pass the arguments dictionary
        )
        logging.info("Language model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading language model or tokenizer: {e}")
        logging.error(traceback.format_exc())
        # Add specific check for accelerate error if modification wasn't applied
        if "requires `accelerate`" in str(e) and DEVICE_MAP == "cpu":
            logging.error(
                "Suggestion: Install accelerate (`pip install accelerate`) or modify code to not pass `device_map` when on CPU."
            )
        return None, None

<<<<<<< HEAD

# Initialize Coqui TTS Engine
=======
# --- (Keep init_coqui_tts as is) ---
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
def init_coqui_tts() -> Optional[TTS]:
    """Initializes the Coqui TTS engine."""
    try:
        logging.info(
            f"Initializing Coqui TTS engine with model: {COQUI_TTS_MODEL_NAME}..."
        )
        # Determine if GPU should be used for TTS
        use_gpu = DEVICE_MAP == "cuda"
        tts_engine = TTS(
            model_name=COQUI_TTS_MODEL_NAME, progress_bar=False, gpu=use_gpu
        )
        logging.info(f"Coqui TTS engine initialized (GPU: {use_gpu}).")

        # Log available speakers/languages if applicable
        if (
            tts_engine.is_multi_speaker
            and hasattr(tts_engine, "speakers")
            and tts_engine.speakers
        ):
            logging.info(f"Available speakers: {tts_engine.speakers}")
            if COQUI_TTS_SPEAKER and COQUI_TTS_SPEAKER not in tts_engine.speakers:
                logging.warning(
                    f"Specified speaker '{COQUI_TTS_SPEAKER}' not found in model. Using default."
                )
        if (
            tts_engine.is_multi_lingual
            and hasattr(tts_engine, "languages")
            and tts_engine.languages
        ):
            logging.info(f"Available languages: {tts_engine.languages}")
            if COQUI_TTS_LANGUAGE and COQUI_TTS_LANGUAGE not in tts_engine.languages:
                logging.warning(
                    f"Specified language '{COQUI_TTS_LANGUAGE}' not found in model. Using default."
                )

        return tts_engine
    except Exception as e:
        logging.error(f"Error initializing Coqui TTS engine: {e}")
        logging.error(traceback.format_exc())
        return None


# --- Core Functions ---

stop_recording = threading.Event()

<<<<<<< HEAD

=======
# --- (Keep listen_for_enter as is) ---
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
def listen_for_enter():
    """Listens for the Enter key press to stop recording."""
    input()  # Wait for Enter key press
    if not stop_recording.is_set():
        print("\nEnter pressed, stopping recording...")
        stop_recording.set()


<<<<<<< HEAD
def recognize_speech(
    audio_model: whisper.Whisper,
    duration: int,
    rate: int,
    language: Optional[str],  # Allow None
=======
# --- (Keep recognize_speech as is) ---
def recognize_speech(
    audio_model: WhisperModel,
    duration: int,
    rate: int,
    language: Optional[str],
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
) -> Optional[str]:
    """Records audio using InputStream and transcribes it using Whisper. Stops early if Enter is pressed."""
    global stop_recording
    stop_recording.clear()

    if not audio_model:
        logging.error("recognize_speech called but Whisper model not loaded.")
        return None

    print(f"\nListening for up to {duration} seconds (press Enter to stop early)...")
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            logging.warning(f"Audio Callback Status: {status}")
        audio_queue.put(indata.copy())

    enter_thread = threading.Thread(target=listen_for_enter, daemon=True)
    enter_thread.start()

    stream = None  # Initialize stream variable
    try:
        # Use sounddevice context manager for cleaner resource handling
        with sd.InputStream(
            samplerate=rate, channels=1, dtype="int16", callback=audio_callback
        ) as stream:
            start_time = time.time()
            while not stop_recording.is_set() and (time.time() - start_time < duration):
                time.sleep(0.1)  # Keep polling interval

            # Ensure the stream is stopped if Enter was pressed or time ran out
            # The context manager handles stream.close()

        logging.debug("Processing recorded audio...")
        recorded_frames = []
        while not audio_queue.empty():
            try:
                recorded_frames.append(audio_queue.get_nowait())
            except queue.Empty:
                break

        if not recorded_frames:
            logging.warning("No audio frames were captured.")
            return None
        else:
            logging.debug(f"Captured {len(recorded_frames)} audio blocks.")

        audio_data_np = np.concatenate(recorded_frames, axis=0)
        logging.debug(f"Total audio samples: {len(audio_data_np)}")

        max_amplitude = np.max(np.abs(audio_data_np))
        logging.debug(f"Max audio amplitude (int16): {max_amplitude}")
<<<<<<< HEAD
        if max_amplitude < 500:
            logging.warning("Warning - Recorded audio seems very quiet.")

        # Normalize audio for Whisper
        audio_float32 = audio_data_np.astype(np.float32) / 32768.0

        logging.info("Transcribing audio with Whisper...")
        # Determine Whisper options
        whisper_options = {
            "language": language if language and language.lower() != "none" else None,
            "fp16": DEVICE_MAP == "cuda",  # Use fp16 only if on GPU
            # "verbose": True, # Uncomment for detailed Whisper logs
        }
        logging.debug(f"Whisper options: {whisper_options}")

        result = audio_model.transcribe(audio_float32.flatten(), **whisper_options)

        logging.debug(f"Raw Whisper Result: {result}")

        text = result["text"].strip()
        if not text:
            logging.warning("Whisper transcribed empty text.")
=======
        if (
            max_amplitude < 500
        ):  # Threshold might need tuning depending on mic sensitivity
            logging.warning("Warning - Recorded audio seems very quiet.")

        # Normalize audio for Whisper (remains the same)
        audio_float32 = audio_data_np.astype(np.float32) / 32768.0

        logging.info("Transcribing audio with Faster Whisper...")
        # Determine Whisper options
        whisper_options = {
            "language": language if language and language.lower() != "none" else None,
            # Add other faster-whisper specific options if needed (e.g., beam_size)
            # "vad_filter": True, # faster-whisper has built-in VAD filtering
        }
        logging.debug(f"Faster Whisper options: {whisper_options}")

        # Transcribe using faster-whisper
        # The transcribe method returns an iterator of segments and transcription info
        segments, info = audio_model.transcribe(
            audio_float32.flatten(), **whisper_options
        )

        # Concatenate text from all segments
        full_text = "".join(segment.text for segment in segments)
        text = full_text.strip()

        # Log detected language if needed
        if info:
            logging.debug(
                f"Detected language: {info.language} with probability {info.language_probability}"
            )

        logging.debug(f"Raw Faster Whisper Text: {text}")

        if not text:
            logging.warning("Faster Whisper transcribed empty text.")
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
            return None

        print(f"Recognized: '{text}'")
        return text

    except sd.PortAudioError as e:
        logging.error(f"Audio device error: {e}. Is a microphone connected/configured?")
        return None
    except Exception as e:
        logging.error(f"Error during speech recognition: {e}")
        logging.error(traceback.format_exc())
        return None
    finally:
        # Ensure the event is set even if an error occurs before the loop finishes
        if not stop_recording.is_set():
            stop_recording.set()
        # Wait for the input thread to potentially finish
        if enter_thread.is_alive():
<<<<<<< HEAD
            enter_thread.join(timeout=0.5)  # Give it a moment to exit cleanly


=======
            # Give it a moment to exit cleanly after input() potentially blocks
            enter_thread.join(timeout=0.5)


# --- (Keep generate_response as is) ---
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
def generate_response(
    prompt: str,
    llm_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    history: list[dict[str, str]],
) -> Optional[str]:
    """Generates a response using the language model based on the user prompt and history."""
    if not llm_model or not tokenizer:
        logging.error("Language model or tokenizer not loaded.")
        return None
    if not isinstance(prompt, str) or not prompt.strip():
        logging.warning("Received empty or invalid prompt.")
        return None

    try:
<<<<<<< HEAD
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt.strip()})
=======
        # Construct messages including system prompt and history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)  # Add user/assistant turns
        messages.append(
            {"role": "user", "content": prompt.strip()}
        )  # Add current user prompt
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718

        # Ensure pad_token_id is set correctly before generation
        current_pad_token_id = tokenizer.pad_token_id
        if current_pad_token_id is None and tokenizer.eos_token_id is not None:
            logging.debug(
                "Temporarily setting pad_token_id to eos_token_id for generation."
            )
            current_pad_token_id = tokenizer.eos_token_id
        elif current_pad_token_id is None:
            logging.error(
                "Cannot generate response: Both pad_token_id and eos_token_id are None."
            )
            return None

        # Tokenize using the chat template
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(llm_model.device)

        logging.info("Generating response...")
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = llm_model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                pad_token_id=current_pad_token_id,  # Use the determined pad token id
                eos_token_id=tokenizer.eos_token_id,  # Use the tokenizer's EOS token ID
            )

<<<<<<< HEAD
=======
        # Decode only the newly generated tokens
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
        response_tokens = outputs[0][inputs.shape[-1] :]
        response_text = tokenizer.decode(
            response_tokens, skip_special_tokens=True
        ).strip()

        print(f"Generated Response: '{response_text}'")
        return response_text if response_text else None

    except (RuntimeError, ValueError) as e:
        logging.error(f"Error during text generation: {e}")
        if "CUDA out of memory" in str(e):
            logging.warning(
                "Suggestion: Try a smaller LLM, reduce MAX_NEW_TOKENS/history, or use CPU."
            )
        logging.error(traceback.format_exc())
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during generation: {repr(e)}")
        logging.error(traceback.format_exc())
        return None

<<<<<<< HEAD

=======
# --- (Keep speak as is) ---
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
def speak(text: str, tts_engine: TTS):
    """Uses the Coqui TTS engine to speak the given text and plays it."""
    if not tts_engine:
        logging.error("Coqui TTS engine not initialized.")
        return
    if not isinstance(text, str) or not text.strip():
        logging.warning("Received empty text to speak.")
        return

    try:
        logging.info("Synthesizing speech with Coqui TTS...")

        # Determine speaker and language based on config and model capabilities
        speaker_arg = COQUI_TTS_SPEAKER if tts_engine.is_multi_speaker else None
        language_arg = COQUI_TTS_LANGUAGE if tts_engine.is_multi_lingual else None

        # Synthesize speech to a file
        tts_engine.tts_to_file(
            text=text,
            speaker=speaker_arg,
            language=language_arg,
            file_path=TTS_OUTPUT_FILENAME,
        )
        logging.info(f"Speech synthesized to {TTS_OUTPUT_FILENAME}")

        # Play the generated audio file
        logging.info("Playing synthesized speech...")
        audio_data, samplerate = sf.read(TTS_OUTPUT_FILENAME, dtype="float32")
        sd.play(audio_data, samplerate)
        sd.wait()  # Wait until playback is finished
        logging.info("Speaking finished.")

        # Optional: Clean up the temporary audio file
        try:
            os.remove(TTS_OUTPUT_FILENAME)
            logging.debug(f"Removed temporary TTS file: {TTS_OUTPUT_FILENAME}")
        except OSError as e:
            logging.warning(
                f"Could not remove temporary TTS file {TTS_OUTPUT_FILENAME}: {e}"
            )

    except RuntimeError as e:
        # Catch potential runtime errors from TTS (e.g., Cuda errors)
        logging.error(f"Runtime error during Coqui TTS synthesis or playback: {e}")
        logging.error(traceback.format_exc())
        if "CUDA out of memory" in str(e):
            logging.warning(
                "Suggestion: Try using a less demanding TTS model or run TTS on CPU."
            )
    except Exception as e:
        logging.error(f"Error during text-to-speech synthesis or playback: {e}")
        logging.error(traceback.format_exc())


# --- Main Execution ---
if __name__ == "__main__":
<<<<<<< HEAD
    if not check_ffmpeg():
        exit(1)

    whisper_model = load_whisper_model(WHISPER_MODEL_SIZE)
    llm_model, tokenizer = load_language_model(MODEL_NAME)
    # Initialize Coqui TTS instead of pyttsx3
    tts_synthesizer = init_coqui_tts()  # Renamed variable

    # Check all components loaded successfully
    if not whisper_model or not llm_model or not tokenizer or not tts_synthesizer:
        print("\nInitialization of one or more components failed. Exiting.")
        # Explicitly log which component failed if possible
        if not whisper_model:
            logging.error("Whisper model failed to load.")
        if not llm_model or not tokenizer:
            logging.error("Language model or tokenizer failed to load.")
        if not tts_synthesizer:
            logging.error("Coqui TTS engine failed to load.")
=======

    whisper_model = load_whisper_model(WHISPER_MODEL_SIZE)
    llm_model, tokenizer = load_language_model(MODEL_NAME)
    tts_synthesizer = init_coqui_tts()

    if not whisper_model or not llm_model or not tokenizer or not tts_synthesizer:
        print("\nInitialization of one or more components failed. Exiting.")
        # Consider adding more specific logging here about which component failed if needed
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
        exit(1)

    print("\nInitialization complete. Starting interaction loop (Ctrl+C to exit).")

    conversation_history: list[dict[str, str]] = []

    try:
        while True:
            # 1. Recognize Speech
            user_command = recognize_speech(
                whisper_model,
                duration=RECORD_SECONDS,
                rate=SAMPLE_RATE,
                language=WHISPER_LANGUAGE,
            )

            if user_command:
<<<<<<< HEAD
                conversation_history.append({"role": "user", "content": user_command})

                # 2. Generate Response
                ai_response = generate_response(
                    user_command, llm_model, tokenizer, conversation_history
                )

                if ai_response:
                    conversation_history.append(
                        {"role": "assistant", "content": ai_response}
                    )
                    # Trim history
                    if (
                        len(conversation_history) > MAX_HISTORY_LENGTH * 2
                    ):  # Keep user+assistant pairs
                        # Keep system prompt + last N turns (user+assistant)
                        keep_turns = MAX_HISTORY_LENGTH
                        conversation_history = conversation_history[-(keep_turns * 2) :]
                        logging.debug(f"History trimmed to last {keep_turns} turns.")

                    # 3. Speak Response using Coqui TTS
                    speak(ai_response, tts_synthesizer)  # Pass the coqui engine
=======
                # Append user message before generation
                current_user_turn = {"role": "user", "content": user_command}

                # Prepare history for the LLM (don't modify the main history list yet)
                history_for_llm = conversation_history + [current_user_turn]

                # Trim history *before* sending to LLM if it exceeds the limit
                # Keep the last N turns (user + assistant pairs)
                if len(history_for_llm) > MAX_HISTORY_LENGTH * 2:
                    keep_items = MAX_HISTORY_LENGTH * 2
                    history_for_llm = history_for_llm[-keep_items:]
                    logging.debug(
                        f"History for LLM trimmed to last {MAX_HISTORY_LENGTH} turns."
                    )

                # 2. Generate Response
                ai_response = generate_response(
                    user_command,
                    llm_model,
                    tokenizer,
                    history_for_llm,  # Pass the potentially trimmed history
                )

                if ai_response:
                    # Now add the user turn and the successful assistant response to the main history
                    conversation_history.append(current_user_turn)
                    conversation_history.append(
                        {"role": "assistant", "content": ai_response}
                    )

                    # Trim the main conversation_history list permanently if needed
                    # (This ensures the list doesn't grow indefinitely, separate from LLM input trimming)
                    if (
                        len(conversation_history) > MAX_HISTORY_LENGTH * 2 + 5
                    ):  # Add buffer
                        keep_items = MAX_HISTORY_LENGTH * 2
                        conversation_history = conversation_history[-keep_items:]
                        logging.debug(f"Main history list trimmed.")

                    # 3. Speak Response using Coqui TTS
                    speak(ai_response, tts_synthesizer)
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718
                else:
                    print(
                        "No response generated or an error occurred during generation."
                    )
<<<<<<< HEAD
                    # Optionally speak an error message
                    # speak("Sorry, I couldn't generate a response.", tts_synthesizer)

            else:
                print("No command recognized or an error occurred during recognition.")
                # Optionally speak a message indicating no input was heard
                # speak("I didn't hear anything.", tts_synthesizer)
=======
                    # Don't add user message to history if generation failed
                    speak("Sorry, I couldn't generate a response.", tts_synthesizer)

            else:
                print("No command recognized or an error occurred during recognition.")
                # Speak a message indicating no input was heard
                speak(
                    "I didn't hear anything clearly. Please try again.", tts_synthesizer
                )  # Slightly more informative
>>>>>>> e7c1b2840d3967661210ea1c2c97a391a6902718

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting program.")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred in the main loop: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Cleanup: Remove temporary TTS file if it still exists
        if os.path.exists(TTS_OUTPUT_FILENAME):
            try:
                os.remove(TTS_OUTPUT_FILENAME)
                logging.info(f"Cleaned up temporary TTS file: {TTS_OUTPUT_FILENAME}")
            except OSError as e:
                logging.warning(
                    f"Could not remove temporary TTS file {TTS_OUTPUT_FILENAME} on exit: {e}"
                )

        # No explicit stop needed for Coqui TTS or sounddevice playback usually
        print("Program finished.")
