import queue  # Import queue for thread-safe data exchange between audio callback and processing.
import numpy as np  # Import numpy for numerical operations, especially audio arrays.
import logging  # Import logging for event logging.
from typing import (  # Import typing for type hints.
    Optional,  # Optional type hint.
    Tuple,  # Tuple type hint.
    List,  # List type hint.
    Dict,  # Dict type hint.
    Any,  # Any type hint.
)  # End typing imports.
import librosa # Import librosa for resampling
import threading  # Import threading (though asyncio.to_thread is primarily used).
import traceback  # Import traceback for formatting exception information.
import asyncio  # Import asyncio for asynchronous operations
import os  # Import os for environment variables
import string # Import string for character manipulation

# Optional imports - attempt to import and handle errors gracefully.
try:  # Start try block for essential imports.
    import sounddevice as sd  # Import sounddevice for audio recording and playback.
    from faster_whisper import (  # Import from faster_whisper.
        WhisperModel,  # Import WhisperModel class.
    )  # End faster_whisper imports.
    import torch  # Import torch for PyTorch tensor operations (used by models).
    from transformers import (  # Import from transformers.
        AutoTokenizer,  # Import AutoTokenizer class.
        AutoModelForCausalLM,  # Import standard AutoModelForCausalLM.
    )  # End transformers imports.
    from TTS.api import TTS  # Import TTS API from Coqui TTS.
    import av  # Import av for potential FFmpeg dependency handling (though TTS might handle it internally).
except ImportError as e:  # Catch import errors.
    # Print detailed error message and guidance if imports fail.
    print(f"Error importing required library: {e}")  # Print the specific import error.
    print(  # Provide guidance on missing dependencies.
        "Please ensure sounddevice, faster-whisper, torch, transformers, TTS (coqui-tts), numpy, asyncio, and av are installed."  # List required libraries.
    )  # End print guidance.
    print(  # Add notes about potential system dependencies.
        "You might also need FFmpeg installed (often provided by 'av') and espeak-ng (for some TTS models)."  # Mention system dependencies.
    )  # End print notes.
    print( # Add Raspberry Pi specific notes
        "On Raspberry Pi, ensure you are using a 64-bit OS for optimal performance."
    )
    exit(1)  # Exit the script if essential libraries are missing.

# --- Configuration ---
# NOTE: Raspberry Pi 5 Performance Considerations:
# - LLM_MODEL_NAME: Qwen2.5-0.5B is small, but LLM inference is the most demanding task.
#   Consider even smaller models or quantized formats (e.g., GGUF with ctransformers/llama-cpp-python, or ONNX with optimum)
#   for significant speedups, though this requires code changes to load/run them.
# - WHISPER_MODEL_SIZE: 'tiny' or 'base' will be faster than 'small' but less accurate. 'small' with 'int8' is a reasonable balance.
# - COQUI_TTS_MODEL_NAME: VITS models can be demanding. Consider lighter TTS alternatives if needed.
# - Ensure adequate cooling for the Pi 5, as sustained load can cause thermal throttling.

LLM_MODEL_NAME: str = (
    "unsloth/Qwen2.5-0.5B"  # Define the Hugging Face model name for the LLM.
)
WHISPER_MODEL_SIZE: str = (
    "small"  # Faster Whisper model size (tiny, base, small, medium, large). 'small' recommended balance for Pi.
)
WHISPER_LANGUAGE: Optional[str] = (  # Target language for Whisper ('en', or None for auto-detect).
    "en"
)
SAMPLE_RATE: int = (
    48000  # Audio sample rate (Hz). Try 48kHz as input device might support it.
)
RECORD_SECONDS: int = (
    60  # Max recording duration (seconds).
)
MIN_NEW_TOKENS: int = (
    16  # Minimum LLM generation length.
)
MAX_NEW_TOKENS: int = (
    # Max LLM generation length. Reduce this if hitting memory limits or for faster responses on Pi.
    512
)
# Force CPU execution for Raspberry Pi
DEVICE_MAP: str = "cpu"
TORCH_DTYPE = torch.float32 # Use float32 for CPU inference.

# Set PyTorch intra-op parallelism threads (adjust based on Pi 5 cores and testing)
# Often letting PyTorch decide is fine, but explicit setting can sometimes help.
# os.environ["OMP_NUM_THREADS"] = "4" # Example: Set to number of physical cores
# torch.set_num_threads(4) # Example: Set PyTorch specific thread count

# --- Optimized System Prompt ---
# Aims for general knowledge but specialized in survival, with a conversational tone.
SYSTEM_PROMPT: str = (
    "You're a knowledgeable AI assistant, ready to chat about almost anything. "
    "However, you have deep expertise in survival situations – think first aid, finding resources, "
    "shelter, navigation, the works. When survival topics come up, provide clear, practical, "
    "and actionable advice like you're guiding someone through it. Keep it conversational and direct."
)

# --- Characters to remove before TTS ---
CHARS_TO_REMOVE_FOR_TTS: str = "!@#$%^&*()_+-="
TTS_TRANSLATION_TABLE = str.maketrans('', '', CHARS_TO_REMOVE_FOR_TTS)

# --- Coqui TTS Configuration ---
COQUI_TTS_MODEL_NAME: str = (
    "tts_models/en/ljspeech/glow-tts"  # Lighter Glow-TTS model for potentially faster inference.
)
COQUI_TTS_SPEAKER: Optional[str] = (
    "p225"  # Speaker ID for multi-speaker models (e.g., VCTK).
)
COQUI_TTS_SPEAKER_WAV: Optional[str] = (
    None  # Path to reference WAV for speaker cloning (XTTS models).
)
COQUI_TTS_LANGUAGE: Optional[str] = (
    "en"  # Language code for multi-lingual TTS models.
)

MAX_HISTORY_LENGTH: int = (
    3  # Max conversation turns (user + assistant pairs) in history. Keep low on Pi.
)
LOG_LEVEL = (
    logging.INFO
)  # Logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL).
# LOG_LEVEL = logging.DEBUG # Uncomment for detailed diagnostic logs.

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)

# --- Global Async Event for Recording Control ---
stop_recording_event = asyncio.Event()

# --- Model Loading Functions (Synchronous - executed once at startup) ---

def load_whisper_model(
    model_size: str, device: str
) -> Optional[WhisperModel]:
    """Loads the specified Faster Whisper model synchronously."""
    try:
        logging.info(f"Loading Faster Whisper model ({model_size}) for {device}...")
        # Use 'int8' compute type for CPU (Raspberry Pi) for better performance.
        # Use 'float16' or 'float32' if running on a CUDA GPU elsewhere.
        compute_type = "int8" if device == "cpu" else "float16"
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            # Consider limiting CPU threads if needed, though default is often fine
            # cpu_threads=4, # Example: Limit threads used by Whisper internal ops
            # num_workers=1, # Example: Limit parallel processing workers
        )
        logging.info(f"Faster Whisper model loaded to {device} (compute: {compute_type}).")
        return model
    except Exception as e:
        logging.error(f"Error loading Faster Whisper model: {e}\n{traceback.format_exc()}")
        return None

def load_language_model(
    model_name: str, device: str, torch_dtype: torch.dtype
) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """Loads the Hugging Face tokenizer and language model synchronously."""
    model = None
    tokenizer = None
    try:
        # --- Load Tokenizer ---
        logging.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logging.info("Tokenizer loaded.")

        # --- Ensure chat template and pad token exist ---
        if tokenizer.chat_template is None:
            logging.warning("Tokenizer missing chat_template. Applying a default template.")
            # Define a default chat template string (adjust if needed for the specific model)
            default_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )
            tokenizer.chat_template = default_template
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                logging.info("Tokenizer missing pad_token. Setting pad_token to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                logging.warning("Tokenizer missing both pad_token and eos_token. Adding default tokens: '<|pad|>' and '<|endoftext|>'." )
                tokenizer.add_special_tokens({"pad_token": "<|pad|>", "eos_token": "<|endoftext|>"})
                # NOTE: Adding tokens might require resizing model embeddings if not already handled.
                # This is less common with pre-trained models but good to be aware of.

        # --- Load Model ---
        logging.info(f"Loading language model {model_name} with Transformers to {device} ({torch_dtype})...")
        # NOTE: For Raspberry Pi (CPU), consider memory usage.
        # `low_cpu_mem_usage=True` might help during loading for larger models,
        # though less critical for a 0.5B parameter model.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            # low_cpu_mem_usage=True, # Potentially useful on low-RAM systems during load
            # device_map=device # Explicitly set device map
        )

        # Explicitly move model to the target device
        model.to(device)
        logging.info(f"Language model {model_name} loaded successfully to {model.device}.")

        # --- Model Optimization (Optional but Recommended for Pi) ---
        # For CPU inference, torch.compile might offer speedups on supported models/ops,
        # but can increase startup time and memory usage. Requires PyTorch 2.0+.
        # try:
        #     # mode options: "default", "reduce-overhead", "max-autotune"
        #     # 'reduce-overhead' is often good for inference.
        #     model = torch.compile(model, mode="reduce-overhead", backend="inductor")
        #     logging.info("Applied torch.compile to the language model for potential speedup.")
        # except Exception as compile_e:
        #     logging.warning(f"Could not apply torch.compile: {compile_e}")

        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading language model or tokenizer: {e}\n{traceback.format_exc()}")
        return None, None

def init_coqui_tts(device: str) -> Optional[TTS]:
    """Initializes the Coqui TTS engine synchronously."""
    try:
        logging.info(f"Initializing Coqui TTS engine: {COQUI_TTS_MODEL_NAME}...")
        # Force CPU for TTS on Raspberry Pi, even if GPU was detected elsewhere (unlikely)
        use_gpu = False # Explicitly set to False for Pi
        if device != "cpu":
             logging.warning(f"Main device is {device}, but forcing CPU for Coqui TTS.")

        tts_engine = TTS(
            model_name=COQUI_TTS_MODEL_NAME,
            progress_bar=False,
            gpu=use_gpu, # Use the determined CPU setting
        )
        logging.info(f"Coqui TTS engine initialized (GPU: {use_gpu}).")

        # Log speaker/language info if available
        if tts_engine.is_multi_speaker and hasattr(tts_engine, "speakers") and tts_engine.speakers:
            logging.debug(f"Available TTS speakers: {tts_engine.speakers}")
            if COQUI_TTS_SPEAKER and COQUI_TTS_SPEAKER not in tts_engine.speakers:
                logging.warning(f"Specified speaker '{COQUI_TTS_SPEAKER}' not found in model '{COQUI_TTS_MODEL_NAME}'. Using default.")
        if tts_engine.is_multi_lingual and hasattr(tts_engine, "languages") and tts_engine.languages:
            logging.debug(f"Available TTS languages: {tts_engine.languages}")
            if COQUI_TTS_LANGUAGE and COQUI_TTS_LANGUAGE not in tts_engine.languages:
                logging.warning(f"Specified language '{COQUI_TTS_LANGUAGE}' not found in model '{COQUI_TTS_MODEL_NAME}'. Using default.")

        return tts_engine
    except Exception as e:
        logging.error(f"Error initializing Coqui TTS engine: {e}\n{traceback.format_exc()}")
        return None

# --- Asynchronous Core Functions ---

async def listen_for_enter_async():
    """Runs blocking input() in a separate thread to signal recording stop via an event."""
    global stop_recording_event
    logging.debug("Enter listener thread started.")
    await asyncio.to_thread(input) # Wait for Enter key press
    if not stop_recording_event.is_set():
        print("\nEnter pressed, stopping recording...")
        stop_recording_event.set()
    logging.debug("Enter listener thread finished.")

async def recognize_speech_async(
    audio_model: WhisperModel,
    duration: int,
    rate: int,
    language: Optional[str],
) -> Optional[str]:
    """Records audio asynchronously, handles early stop, and transcribes using Whisper in a thread."""
    global stop_recording_event
    stop_recording_event.clear()

    if not audio_model:
        logging.error("recognize_speech_async called but Whisper model not loaded.")
        return None

    print(f"\nListening for up to {duration} seconds (press Enter to stop early)...")
    audio_queue = queue.Queue()

    # --- Sounddevice Callback (runs in separate audio thread) ---
    def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags):
        """Callback function called by sounddevice for each new audio buffer."""
        if status:
            logging.warning(f"Audio Callback Status: {status}")
        # Put a *copy* of the incoming audio data buffer onto the queue.
        audio_queue.put(indata.copy())

    # --- Start Listener and Recording ---
    enter_listener_task = asyncio.create_task(listen_for_enter_async())
    stream = None
    audio_data_np = None
    recorded_frames = [] # Keep frames list outside try block

    try:
        # Use context manager for sounddevice InputStream
        with sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="int16", # Common format, check if Whisper prefers float32 directly
            callback=audio_callback,
        ) as stream:
            start_time = asyncio.get_event_loop().time()
            while not stop_recording_event.is_set():
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - start_time
                if elapsed_time >= duration:
                    logging.debug(f"Recording duration limit ({duration}s) reached.")
                    if not stop_recording_event.is_set():
                        stop_recording_event.set()
                    break # Exit loop
                await asyncio.sleep(0.1) # Yield control

        # Recording stopped (Enter or timeout)
        logging.debug("Recording stream stopped.")

        # Process remaining audio chunks from queue
        logging.debug("Processing recorded audio chunks from queue...")
        while not audio_queue.empty():
            try:
                frame = audio_queue.get_nowait()
                recorded_frames.append(frame)
            except queue.Empty:
                logging.debug("Audio queue empty during final retrieval.")
                break

        if not recorded_frames:
            logging.warning("No audio frames were captured during recording.")
            # Ensure listener task is cleaned up even if no audio
            if not enter_listener_task.done():
                enter_listener_task.cancel()
                try: await enter_listener_task
                except asyncio.CancelledError: pass
            return None

        # Concatenate audio frames
        audio_data_np = np.concatenate(recorded_frames, axis=0)
        logging.debug(f"Concatenated {len(recorded_frames)} audio blocks. Shape: {audio_data_np.shape}")

        # --- Audio Quality Check (Optional) ---
        max_amplitude = np.max(np.abs(audio_data_np)) if audio_data_np.size > 0 else 0
        logging.debug(f"Max audio amplitude (int16 range): {max_amplitude}")
        amplitude_threshold = 500
        if max_amplitude < amplitude_threshold:
            logging.warning(f"Recorded audio seems quiet (max amplitude: {max_amplitude} < {amplitude_threshold}). Check mic levels.")

        # --- Normalize and Transcribe ---
        # Convert int16 to float32 and normalize for Whisper
        audio_float32 = audio_data_np.astype(np.float32) / 32768.0

        logging.info("Transcribing audio using Faster Whisper (running in thread)...")
        whisper_options = {
            "language": language if language and language.lower() != "none" else None,
            # Add other Whisper options if needed: beam_size, temperature, vad_filter=True etc.
            # VAD (Voice Activity Detection) can help filter silence on Pi
            # "vad_filter": True,
            # "vad_parameters": dict(min_silence_duration_ms=500),
        }
        logging.debug(f"Passing options to Faster Whisper transcribe: {whisper_options}")


        # Run blocking Whisper transcription in a separate thread
        segments, info = await asyncio.to_thread(
            audio_model.transcribe,
            audio_float32.flatten(), # Ensure 1D array
            **whisper_options,
        )

        # Combine transcribed segments
        full_text = "".join(segment.text for segment in segments)
        text = full_text.strip()

        if info and hasattr(info, "language") and hasattr(info, "language_probability"):
            logging.debug(f"Whisper detected language: {info.language} (prob: {info.language_probability:.2f})")

        logging.debug(f"Raw transcribed text: '{text}'")

        if not text:
            logging.warning("Whisper transcription resulted in empty text.")
            return None

        print(f"Recognized: '{text}'")
        return text

    except sd.PortAudioError as e:
        logging.error(f"Sounddevice/PortAudio error: {e}. Check microphone configuration.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during speech recognition: {e}\n{traceback.format_exc()}")
        return None
    finally:
        # Ensure stop event is set and listener task is cancelled cleanly
        if not stop_recording_event.is_set():
            stop_recording_event.set()
        if 'enter_listener_task' in locals() and enter_listener_task and not enter_listener_task.done():
            enter_listener_task.cancel()
            try:
                await enter_listener_task
            except asyncio.CancelledError:
                logging.debug("Enter listener task cancelled during cleanup.")
            except Exception as final_cancel_e:
                logging.error(f"Error awaiting cancelled enter_listener_task in finally: {final_cancel_e}")


async def generate_response_async(
    prompt: str,
    llm_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    history: List[Dict[str, str]],
) -> Optional[str]:
    """Generates LLM response using model.generate in a separate thread."""
    if not llm_model or not tokenizer or not torch:
        logging.error("generate_response_async called but LLM/tokenizer/torch not available.")
        return None
    if not isinstance(prompt, str) or not prompt.strip():
        logging.warning("Received empty or invalid prompt for generation.")
        return None

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt.strip()})

        # --- Ensure pad_token_id is set ---
        current_pad_token_id = tokenizer.pad_token_id
        if current_pad_token_id is None and tokenizer.eos_token_id is not None:
            logging.debug("Using eos_token_id as pad_token_id for generation.")
            current_pad_token_id = tokenizer.eos_token_id
        elif current_pad_token_id is None:
            logging.error("Cannot generate: Both pad_token_id and eos_token_id are None.")
            return None

        # --- Prepare inputs (Tokenization) ---
        # This part is synchronous but usually fast
        formatted_prompt_string = None
        tokenized_inputs = None
        input_ids = None
        attention_mask = None
        try:
            # Apply chat template to get formatted string
            formatted_prompt_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Tokenize the formatted string
            # No padding needed for single sequence generation usually
            tokenized_inputs = tokenizer(
                formatted_prompt_string,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(llm_model.device) # Move inputs to the model's device (CPU for Pi)

            input_ids = tokenized_inputs.get("input_ids")
            attention_mask = tokenized_inputs.get("attention_mask")

            if input_ids is None:
                logging.error("Tokenization failed to produce 'input_ids'.")
                return None
            if attention_mask is None:
                logging.warning("Attention mask missing. Creating default mask.")
                attention_mask = torch.ones_like(input_ids, device=llm_model.device)

        except Exception as e:
            logging.error(f"Error during input formatting/tokenization: {e}\n{traceback.format_exc()}")
            return None

        input_length = input_ids.shape[-1]
        logging.debug(f"Input sequence length: {input_length} tokens.")

        # --- Generate Response (Run blocking model.generate in thread) ---
        logging.info(f"Generating LLM response ({input_length} tokens -> min:{MIN_NEW_TOKENS}, max:{MAX_NEW_TOKENS}) (running in thread)...")

        @torch.no_grad() # Crucial for inference performance and memory
        def _generate_sync():
            """Inner function to run model.generate synchronously."""
            # Use generate method
            outputs = llm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                min_new_tokens=MIN_NEW_TOKENS,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7, # Adjust creativity/randomness
                do_sample=True, # Enable sampling for temperature > 0
                pad_token_id=current_pad_token_id, # Use determined pad token id
                eos_token_id=tokenizer.eos_token_id, # Allow model to stop naturally
            )
            return outputs

        outputs_tensor = await asyncio.to_thread(_generate_sync)

        # --- Decode response ---
        # Extract only the newly generated tokens
        response_tokens = outputs_tensor[0][input_length:]
        response_text = tokenizer.decode(
            response_tokens,
            skip_special_tokens=True
        ).strip()

        logging.info(f"LLM generation complete. Output length: {len(response_tokens)} tokens.")
        print(f"Generated Response: '{response_text[:150]}...'") # Print truncated response

        return response_text if response_text else None

    except (RuntimeError, ValueError) as e:
        logging.error(f"Error during text generation execution: {e}")
        if "CUDA out of memory" in str(e): # Less likely on Pi, but good practice
            logging.warning("CUDA OOM Error during generation. Reduce MAX_NEW_TOKENS or use smaller model.")
        logging.error(traceback.format_exc())
        return None
    except Exception as e:
        logging.error(f"Unexpected error during generation: {repr(e)}\n{traceback.format_exc()}")
        return None

# --- Helper function for TTS playback ---
def _play_audio_blocking(audio_data: np.ndarray, sample_rate: int):
    """Plays audio using sounddevice.play() and waits synchronously. Resamples if needed."""
    try:
        target_sr = 48000 # Choose a widely supported target sample rate (e.g., 44100 or 48000)
        playback_data = audio_data

        # Resample if the original TTS rate doesn't match the target playback rate
        if sample_rate != target_sr:
            logging.info(f"Resampling audio for playback from {sample_rate} Hz to {target_sr} Hz...")
            # Ensure audio_data is 1D float for librosa resampling
            playback_data = playback_data.flatten().astype(np.float32)
            playback_data = librosa.resample(playback_data, orig_sr=sample_rate, target_sr=target_sr)
            logging.info("Resampling complete.")
            current_sample_rate = target_sr
        else:
            current_sample_rate = sample_rate

        logging.debug(f"Starting playback of {len(playback_data)} samples at {current_sample_rate} Hz.")
        sd.play(playback_data, current_sample_rate)
        sd.wait() # Block until playback is finished
        logging.debug("Playback finished.")
    except (sd.PortAudioError, Exception) as e:
        logging.error(f"Error during audio playback (_play_audio_blocking): {e}\n{traceback.format_exc()}")

async def speak_async(text: str, tts_engine: TTS):
    """Synthesizes speech using Coqui TTS in a thread and plays it back in another thread."""
    if not tts_engine:
        logging.error("speak_async called but Coqui TTS engine not initialized.")
        return
    if not isinstance(text, str) or not text.strip():
        logging.warning("Received empty or invalid text to speak.")
        return

    try:
        logging.info("Synthesizing speech using Coqui TTS (running in thread)...")

        # Determine TTS arguments based on config and model capabilities
        speaker_arg, language_arg, speaker_wav_arg = None, None, None
        tts_model_name_lower = COQUI_TTS_MODEL_NAME.lower()

        if "xtts" in tts_model_name_lower:
            language_arg = COQUI_TTS_LANGUAGE
            speaker_wav_arg = COQUI_TTS_SPEAKER_WAV
            logging.debug(f"Configuring TTS for XTTS (lang: {language_arg}, speaker_wav: {'Provided' if speaker_wav_arg else 'Default'})")
        elif tts_engine.is_multi_speaker:
            speaker_arg = COQUI_TTS_SPEAKER
            logging.debug(f"Configuring TTS for multi-speaker (speaker: {speaker_arg})")

        if tts_engine.is_multi_lingual and not language_arg: # Set language if multi-lingual and not already set (e.g., by XTTS)
             language_arg = COQUI_TTS_LANGUAGE
             logging.debug(f"Configuring TTS for multi-lingual (language: {language_arg})")

        # --- Run blocking TTS synthesis in a separate thread ---
        wav_list = await asyncio.to_thread(
            tts_engine.tts,
            text=text,
            speaker=speaker_arg,
            language=language_arg,
            speaker_wav=speaker_wav_arg,
        )

        if not wav_list:
            logging.error("TTS synthesis returned no audio data.")
            return

        # Convert to numpy float32 array for playback
        audio_data = np.array(wav_list).astype(np.float32)

        # Determine sample rate for playback
        actual_sample_rate = 22050 # Common default for TTS
        try:
            # Try to get the actual rate from the loaded model
            if hasattr(tts_engine, "synthesizer") and hasattr(tts_engine.synthesizer, "output_sample_rate"):
                actual_sample_rate = tts_engine.synthesizer.output_sample_rate
                logging.debug(f"Using actual TTS model output sample rate: {actual_sample_rate} Hz.")
            else:
                 logging.warning(f"Could not determine TTS sample rate from model. Assuming default: {actual_sample_rate} Hz.")
        except Exception as sr_e:
            logging.warning(f"Error retrieving TTS sample rate, assuming default {actual_sample_rate} Hz: {sr_e}")

        logging.info("Playing synthesized speech (running in thread)...")
        # --- Run blocking playback in a separate thread ---
        await asyncio.to_thread(
            _play_audio_blocking,
            audio_data,
            actual_sample_rate,
        )
        logging.info("Finished speaking.")

    except RuntimeError as e:
        logging.error(f"Runtime error during TTS synthesis/playback: {e}")
        if "CUDA out of memory" in str(e): # Unlikely on Pi
            logging.warning("CUDA OOM Error during TTS synthesis.")
        logging.error(traceback.format_exc())
    except Exception as e:
        logging.error(f"Unexpected error during TTS synthesis or playback: {e}\n{traceback.format_exc()}")


# --- Main Asynchronous Execution ---
async def main():
    """Initializes models and runs the main interaction loop."""
    # --- Load Models Synchronously at Startup ---
    # Ensure correct device and dtype are passed based on config
    whisper_model = load_whisper_model(WHISPER_MODEL_SIZE, DEVICE_MAP)
    llm_model, tokenizer = load_language_model(LLM_MODEL_NAME, DEVICE_MAP, TORCH_DTYPE)
    tts_synthesizer = init_coqui_tts(DEVICE_MAP) # Pass device, though it forces CPU inside

    # --- Check Initialization Success ---
    if not all([whisper_model, llm_model, tokenizer, tts_synthesizer]):
        logging.critical("One or more essential components failed to initialize. Cannot continue.")
        print("\nInitialization failed. Check logs. Exiting.")
        return

    print(f"\nInitialization complete. Models loaded to {DEVICE_MAP}. Starting interaction loop (Ctrl+C to exit).")
    conversation_history: List[Dict[str, str]] = []

    try:
        while True:
            # --- 1. Recognize Speech ---
            logging.info("Starting interaction cycle: Listening...")
            user_command = await recognize_speech_async(
                whisper_model,
                RECORD_SECONDS,
                SAMPLE_RATE,
                WHISPER_LANGUAGE,
            )

            # --- 2. Process Command ---
            if user_command:
                logging.info(f"User command recognized: '{user_command}'")
                current_user_turn = {"role": "user", "content": user_command}

                # Prepare history for LLM, trimming if necessary
                history_for_llm = conversation_history + [current_user_turn]
                max_history_items = MAX_HISTORY_LENGTH * 2
                if len(history_for_llm) > max_history_items:
                    history_for_llm = history_for_llm[-max_history_items:]
                    logging.debug(f"History context for LLM trimmed to last {len(history_for_llm)} messages.")

                # --- 3. Generate Response ---
                logging.info("Generating AI response...")
                ai_response = await generate_response_async(
                    user_command, # Pass only current command as prompt
                    llm_model,
                    tokenizer,
                    history_for_llm, # Pass potentially trimmed history
                )

                # --- 4. Process and Speak Response ---
                if ai_response:
                    logging.info(f"AI response generated: '{ai_response[:100]}...'")
                    # Add successful turn to persistent history
                    conversation_history.append(current_user_turn)
                    conversation_history.append({"role": "assistant", "content": ai_response})

                    # Trim persistent history
                    if len(conversation_history) > max_history_items:
                        conversation_history = conversation_history[-max_history_items:]
                        logging.debug(f"Main conversation history trimmed to {len(conversation_history)} messages.")

                    # --- Purge specified characters before TTS ---
                    cleaned_response = ai_response.translate(TTS_TRANSLATION_TABLE)
                    if cleaned_response != ai_response:
                        logging.debug(f"Cleaned response for TTS: '{cleaned_response[:100]}...'")
                    # ---------------------------------------------

                    # --- 5. Speak Response ---
                    logging.info("Speaking AI response...")
                    # Use the cleaned response for TTS
                    await speak_async(cleaned_response, tts_synthesizer)
                else:
                    logging.warning("LLM response generation failed or returned empty.")
                    fallback_msg = "Sorry, I couldn't generate a response for that."
                    print(fallback_msg)
                    # Clean the fallback message too, just in case (though unlikely needed)
                    cleaned_fallback = fallback_msg.translate(TTS_TRANSLATION_TABLE)
                    await speak_async(cleaned_fallback, tts_synthesizer)
            else:
                logging.warning("Speech recognition failed or returned empty text.")
                fallback_msg = "I didn't catch that. Could you please repeat?"
                print(fallback_msg)
                # Clean the fallback message too
                cleaned_fallback = fallback_msg.translate(TTS_TRANSLATION_TABLE)
                await speak_async(cleaned_fallback, tts_synthesizer)

            await asyncio.sleep(0.1) # Small delay to prevent tight loops on errors

    except asyncio.CancelledError:
        logging.info("Main interaction loop cancelled.")
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting program gracefully.")
        logging.info("KeyboardInterrupt received, shutting down.")
    except Exception as e:
        logging.error(f"Unexpected error in main interaction loop: {e}\n{traceback.format_exc()}")
        print(f"\nAn unexpected error occurred: {e}. Check logs. Exiting.")
    finally:
        print("Program finished.")
        logging.info("Main function finished execution.")
        # Optional: Add explicit resource cleanup if needed


if __name__ == "__main__":
    # Ensure script runs within an asyncio event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt during startup/shutdown.")
    except Exception as e:
        logging.critical(f"Critical error during asyncio execution: {e}\n{traceback.format_exc()}")
        print(f"\nA critical error occurred: {e}. Check logs.")
