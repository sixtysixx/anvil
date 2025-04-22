# /home/six/Documents/anvil/SurvAIval/main.py
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

import threading  # Import threading (though asyncio.to_thread is primarily used).
import traceback  # Import traceback for formatting exception information.
import asyncio  # Import asyncio for asynchronous operations.

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
    exit(1)  # Exit the script if essential libraries are missing.

# --- Configuration ---
MODEL_NAME: str = (
    "microsoft/Phi-4-mini-instruct"  # Define the Hugging Face model name for the LLM.
)
WHISPER_MODEL_SIZE: str = (
    "small"  # Define the size of the Faster Whisper model to use (e.g., tiny, base, small, medium, large).
)
WHISPER_LANGUAGE: str = (  # Define the target language for Whisper transcription.
    "en"  # Set to 'en' for English, or None to attempt auto-detection.  # End Whisper language definition.
)
SAMPLE_RATE: int = (
    16000  # Define the audio sample rate (Hz) for recording and Whisper processing.
)
RECORD_SECONDS: int = (
    60  # Define the maximum duration (seconds) for each audio recording session.
)
MAX_NEW_TOKENS: (
    int
) = (  # Define the maximum number of new tokens the LLM can generate in a single response.
    2048  # Set a limit for response length.  # End max new tokens definition.
)
DEVICE_MAP: str = (  # Determine the primary compute device ('cuda' or 'cpu').
    "cuda"  # Default to CUDA if available.
    if torch.cuda.is_available()  # Check if a CUDA-enabled GPU is detected by PyTorch.
    else "cpu"  # Fallback to CPU if CUDA is not available.
)  # End device map definition.
TORCH_DTYPE = (  # Define the torch data type based on the selected device for potential memory/performance optimization.
    torch.float16  # Use float16 (half-precision) if using CUDA.
    if DEVICE_MAP == "cuda"  # Check if the device is CUDA.
    else torch.float32  # Use float32 (single-precision) if using CPU.
)  # End torch dtype definition.

SYSTEM_PROMPT: str = (
    """You are an AI designed to assist with survival in emergency situations. Your expertise includes first aid, finding shelter, securing food and water, and other critical survival skills. Your responses should be clear, concise, and actionable, focused on providing practical advice for individuals in need. Respond like a human would without any formatting like asterisks or hashtags. Act as if you a human giving advice."""  # Define the system prompt to guide the LLM's behavior and persona.  # Set the detailed system prompt string.  # End system prompt definition.
)

# --- Coqui TTS Configuration ---
COQUI_TTS_MODEL_NAME: str = (
    "tts_models/en/vctk/vits"  # Define the default model name for Coqui TTS.
)
COQUI_TTS_SPEAKER: Optional[str] = (
    "p225"  # Define the speaker ID for multi-speaker models like VCTK.  # Set a default speaker ID (check model documentation for available speakers).  # End Coqui TTS speaker definition.
)
COQUI_TTS_SPEAKER_WAV: Optional[str] = (
    None  # Define the path to a reference WAV file for speaker cloning (used by XTTS models).  # Set to None by default, provide a path if using XTTS speaker cloning.  # End Coqui TTS speaker wav definition.
)
COQUI_TTS_LANGUAGE: Optional[
    str
] = (  # Define the language code for multi-lingual TTS models.
    "en"  # Set the default language to English.  # End Coqui TTS language definition.
)

MAX_HISTORY_LENGTH: int = (
    5  # Define the maximum number of conversation turns (user + assistant pairs) to retain in history.
)
LOG_LEVEL = (
    logging.INFO
)  # Define the logging level for standard operation (INFO, DEBUG, WARNING, ERROR, CRITICAL).
# LOG_LEVEL = logging.DEBUG # Uncomment this line for more detailed diagnostic logs.

logging.basicConfig(  # Configure the basic logging settings for the application.
    level=LOG_LEVEL,  # Set the minimum logging level to capture.
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",  # Define the format for log messages.
)  # End logging configuration.

# --- Global Async Event for Recording Control ---
stop_recording_event = (  # Create an asyncio Event object to signal recording stop requests between tasks.
    asyncio.Event()  # Instantiate the event.
)  # End event creation.

# --- Model Loading Functions (Synchronous - executed once at startup) ---


def load_whisper_model(  # Define function to load the Faster Whisper model.
    model_size: str,  # Accept the desired model size (e.g., "small").
) -> Optional[
    WhisperModel
]:  # Return type hint: Optional WhisperModel object or None on failure.
    """Loads the specified Faster Whisper model synchronously."""  # Docstring explaining the function's purpose.
    try:  # Start error handling block for model loading.
        logging.info(  # Log the start of the Whisper model loading process.
            f"Loading Faster Whisper model ({model_size})..."  # Include model size in the log message.
        )  # End logging info.
        compute_type = (  # Determine the compute type (precision) based on the available device.
            "float16"  # Use float16 for potentially faster inference on CUDA.
            if DEVICE_MAP == "cuda"  # Check if the primary device is CUDA.
            else "int8"  # Use int8 for potentially faster inference on CPU for Whisper.
        )  # End compute type determination.
        model = WhisperModel(  # Instantiate the WhisperModel class.
            model_size,  # Pass the specified model size.
            device=DEVICE_MAP,  # Pass the determined device ('cuda' or 'cpu').
            compute_type=compute_type,  # Pass the determined compute type.
        )  # End WhisperModel instantiation.
        logging.info(  # Log successful loading and configuration details.
            f"Faster Whisper model loaded to {DEVICE_MAP} (compute: {compute_type})."  # Include device and compute type.
        )  # End logging info.
        return model  # Return the loaded model object.
    except Exception as e:  # Catch any exceptions that occur during model loading.
        logging.error(  # Log the error with details.
            f"Error loading Faster Whisper model: {e}\n{traceback.format_exc()}"  # Include the exception message and traceback.
        )  # End logging error.
        return None  # Return None to indicate that loading failed.


def load_language_model(  # Define function to load the Hugging Face language model and tokenizer.
    model_name: str,  # Accept the Hugging Face model name (e.g., "microsoft/Phi-4-mini-instruct").
) -> Tuple[  # Return type hint: A tuple containing...
    Optional[
        AutoModelForCausalLM
    ],  # ...an optional language model object (or None on failure).
    Optional[
        AutoTokenizer
    ],  # ...and an optional tokenizer object (or None on failure).
]:  # End return type hint.
    """Loads the Hugging Face tokenizer and language model synchronously."""  # Docstring explaining the function's purpose.
    model = None  # Initialize the model variable to None.
    tokenizer = None  # Initialize the tokenizer variable to None.
    try:  # Start error handling block for loading the tokenizer and model.
        # --- Load Tokenizer ---
        logging.info(  # Log the start of the tokenizer loading process.
            f"Loading tokenizer for {model_name}..."  # Include the model name.
        )  # End logging info.
        tokenizer = AutoTokenizer.from_pretrained(  # Load the tokenizer using the AutoTokenizer class.
            model_name,  # Specify the model name/path.
            trust_remote_code=True,  # Allow loading custom code associated with the model (required for some models like Phi-3).
        )  # End tokenizer loading.
        logging.info("Tokenizer loaded.")  # Log successful tokenizer loading.

        # --- Ensure chat template and pad token exist ---
        if (  # Check if the loaded tokenizer has a predefined chat template.
            tokenizer.chat_template is None  # Access the chat_template attribute.
        ):  # End check.
            logging.warning(  # Log a warning if no chat template is found.
                "Tokenizer missing chat_template. Applying a default template."  # Inform the user that a default will be used.
            )  # End logging warning.
            # Define a default chat template string (example format, adjust if needed for the specific model).
            default_template = (  # Define a generic template structure.
                "{% for message in messages %}"  # Loop through messages.
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"  # Format each message.
                "{% endfor %}"  # End loop.
                "{% if add_generation_prompt %}"  # Check if generation prompt should be added.
                "{{ '<|im_start|>assistant\n' }}"  # Add the assistant prompt marker.
                "{% endif %}"  # End conditional block.
            )  # End default template definition.
            tokenizer.chat_template = default_template  # Apply the default template to the tokenizer instance.  # Assign the template string.  # End template assignment.
        if (  # Check if the tokenizer has a padding token defined.
            tokenizer.pad_token is None  # Access the pad_token attribute.
        ):  # End check.
            if (  # Check if an end-of-sequence (EOS) token exists, which can often be used as a pad token.
                tokenizer.eos_token is not None  # Access the eos_token attribute.
            ):  # End check.
                logging.info(  # Log that the EOS token will be used as the padding token.
                    "Tokenizer missing pad_token. Setting pad_token to eos_token."  # Informative message.
                )  # End logging info.
                tokenizer.pad_token = (  # Assign the EOS token value to the pad_token attribute.
                    tokenizer.eos_token  # Use the existing EOS token.
                )  # End assignment.
            else:  # If neither pad_token nor eos_token exists.
                logging.warning(  # Log a warning that default pad and eos tokens need to be added.
                    "Tokenizer missing both pad_token and eos_token. Adding default tokens: '<|pad|>' and '<|endoftext|>'."  # Specify the tokens being added.
                )  # End logging warning.
                tokenizer.add_special_tokens(  # Add the required special tokens to the tokenizer's vocabulary.
                    {  # Dictionary specifying the tokens to add.
                        "pad_token": "<|pad|>",  # Define the padding token string.
                        "eos_token": "<|endoftext|>",  # Define the end-of-sequence token string.
                    }  # End dictionary.
                )  # End adding special tokens.
        # --- End tokenizer setup ---

        # --- Load Model using standard Transformers ---
        logging.info(  # Log the start of the language model loading process.
            f"Loading language model {model_name} with Transformers to {DEVICE_MAP}..."  # Include model name and target device.
        )  # End logging info.

        model = AutoModelForCausalLM.from_pretrained(  # Load the language model using AutoModelForCausalLM.
            model_name,  # Pass the model name/path.
            torch_dtype=TORCH_DTYPE,  # Specify the desired data type (float16 for CUDA, float32 for CPU).
            trust_remote_code=True,  # Allow loading custom code associated with the model.
            # device_map="auto" # Optionally let transformers handle device placement automatically.
            # Or explicitly set device_map=DEVICE_MAP if needed, though manual .to() is often clearer.
        )  # End AutoModelForCausalLM loading.

        # Explicitly move model to the target device if device_map wasn't used or didn't place it correctly.
        if (  # Check if the model's current device type doesn't match the target device.
            model.device.type
            != DEVICE_MAP  # Compare model's device type with the target device string.
        ):  # End check.
            model.to(
                DEVICE_MAP
            )  # Move the model's parameters and buffers to the target device.
            logging.info(
                f"Model explicitly moved to {DEVICE_MAP}."
            )  # Log the explicit move action.

        logging.info(  # Log successful model loading.
            "Transformers language model loaded successfully."  # Confirmation message.
        )  # End logging info.
        return (
            model,
            tokenizer,
        )  # Return the loaded model and tokenizer objects as a tuple.
    except Exception as e:  # Catch any exceptions during tokenizer or model loading.
        logging.error(  # Log the error with details.
            f"Error loading language model or tokenizer: {e}\n{traceback.format_exc()}"  # Include exception message and traceback.
        )  # End logging error.
        return (
            None,
            None,
        )  # Return None for both model and tokenizer to indicate failure.


def init_coqui_tts() -> (
    Optional[TTS]
):  # Define function to initialize the Coqui TTS engine.
    """Initializes the Coqui TTS engine synchronously."""  # Docstring explaining the function's purpose.
    try:  # Start error handling block for TTS initialization.
        logging.info(  # Log the start of the TTS initialization process.
            f"Initializing Coqui TTS engine: {COQUI_TTS_MODEL_NAME}..."  # Include the configured TTS model name.
        )  # End logging info.
        use_gpu = (  # Determine if the GPU should be used for TTS based on the main DEVICE_MAP setting.
            DEVICE_MAP == "cuda"  # Check if the primary device is CUDA.
        )  # End GPU determination.
        tts_engine = TTS(  # Instantiate the TTS engine from the Coqui TTS library.
            model_name=COQUI_TTS_MODEL_NAME,  # Pass the configured TTS model name.
            progress_bar=False,  # Disable the progress bar during initialization and synthesis.
            gpu=use_gpu,  # Specify whether to use the GPU (True/False).
        )  # End TTS instantiation.
        logging.info(  # Log successful initialization and GPU usage status.
            f"Coqui TTS engine initialized (GPU: {use_gpu})."  # Include GPU status.
        )  # End logging info.

        # Log speaker/language info if available and relevant for the loaded model for debugging/verification.
        if (  # Check if the loaded TTS model supports multiple speakers.
            tts_engine.is_multi_speaker  # Check the multi-speaker flag.
            and hasattr(
                tts_engine, "speakers"
            )  # Check if the 'speakers' attribute exists.
            and tts_engine.speakers  # Check if the speakers list is not empty.
        ):  # End multi-speaker check.
            logging.debug(  # Log the list of available speakers at the DEBUG level.
                f"Available TTS speakers: {tts_engine.speakers}"  # List the speakers.
            )  # End logging debug.
            if (  # Validate if the configured speaker ID exists in the model's speaker list.
                COQUI_TTS_SPEAKER  # Check if a speaker ID is configured.
                and COQUI_TTS_SPEAKER  # Check again (safe).
                not in tts_engine.speakers  # Check if the configured speaker is NOT in the available list.
            ):  # End speaker validation check.
                logging.warning(  # Log a warning if the specified speaker is not found in the model.
                    f"Specified speaker '{COQUI_TTS_SPEAKER}' not found in model '{COQUI_TTS_MODEL_NAME}'. Using default or first available."  # Inform the user.
                )  # End logging warning.
        if (  # Check if the loaded TTS model supports multiple languages.
            tts_engine.is_multi_lingual  # Check the multi-lingual flag.
            and hasattr(  # Check if the 'languages' attribute exists.
                tts_engine, "languages"  # Attribute name.
            )  # End hasattr check.
            and tts_engine.languages  # Check if the languages list is not empty.
        ):  # End multi-lingual check.
            logging.debug(  # Log the list of available languages at the DEBUG level.
                f"Available TTS languages: {tts_engine.languages}"  # List the languages.
            )  # End logging debug.
            if (  # Validate if the configured language code exists in the model's language list.
                COQUI_TTS_LANGUAGE  # Check if a language code is configured.
                and COQUI_TTS_LANGUAGE  # Check again (safe).
                not in tts_engine.languages  # Check if the configured language is NOT in the available list.
            ):  # End language validation check.
                logging.warning(  # Log a warning if the specified language is not found in the model.
                    f"Specified language '{COQUI_TTS_LANGUAGE}' not found in model '{COQUI_TTS_MODEL_NAME}'. Using default or first available."  # Inform the user.
                )  # End logging warning.

        return tts_engine  # Return the initialized TTS engine object.
    except Exception as e:  # Catch any exceptions during TTS initialization.
        logging.error(  # Log the error with details.
            f"Error initializing Coqui TTS engine: {e}\n{traceback.format_exc()}"  # Include exception message and traceback.
        )  # End logging error.
        return None  # Return None to indicate failure.


# --- Asynchronous Core Functions ---


async def listen_for_enter_async():  # Define async function to listen for the Enter key press non-blockingly.
    """Runs blocking input() in a separate thread to signal recording stop via an event."""  # Docstring explaining the function's purpose.
    global stop_recording_event  # Access the global asyncio event object used for signaling.
    logging.debug(  # Log the start of this specific listener task/thread.
        "Enter listener thread started."  # Debug message.
    )  # End logging debug.
    await asyncio.to_thread(  # Run the blocking input() function in a separate thread managed by asyncio's thread pool.
        input  # The blocking function to execute.
        # No arguments needed for input() here.
    )  # End asyncio.to_thread call.
    # This point is reached only after Enter is pressed and input() returns.
    if (  # Check if the recording hasn't already been stopped by other means (e.g., timeout).
        not stop_recording_event.is_set()  # Check the current state of the event.
    ):  # End check.
        print(  # Inform the user that their action is being processed.
            "\nEnter pressed, stopping recording..."  # User feedback message.
        )  # End print.
        stop_recording_event.set()  # Set the event, signaling the recording loop in recognize_speech_async to stop.
    logging.debug(  # Log the completion of this listener task/thread.
        "Enter listener thread finished."  # Debug message.
    )  # End logging debug.


async def recognize_speech_async(  # Define async function for audio recording and speech recognition.
    audio_model: WhisperModel,  # Accept the loaded Whisper model object.
    duration: int,  # Accept the maximum recording duration in seconds.
    rate: int,  # Accept the audio sample rate (e.g., 16000).
    language: Optional[  # Accept the target language code (e.g., "en") or None for auto-detect.
        str
    ],  # Type hint for language.
) -> Optional[  # Return type hint: Optional string containing the transcribed text, or None on failure/no speech.
    str
]:  # End return type hint.
    """Records audio asynchronously, handles early stop via Enter key, and transcribes using Whisper in a thread."""  # Docstring explaining the function's purpose.
    global stop_recording_event  # Access the global asyncio event object.
    stop_recording_event.clear()  # Reset the event to False at the beginning of each new recording session.

    if (
        not audio_model
    ):  # Check if the Whisper model object is valid (was loaded successfully).
        logging.error(  # Log an error if the model is missing.
            "recognize_speech_async called but Whisper model not loaded."  # Error message.
        )  # End logging error.
        return None  # Return None as transcription cannot proceed without the model.

    print(  # Prompt the user indicating that recording is starting.
        f"\nListening for up to {duration} seconds (press Enter to stop early)..."  # Inform about duration and the early stop mechanism.
    )  # End print prompt.
    audio_queue = (  # Create a thread-safe queue to pass audio chunks from the sounddevice callback to the main async task.
        queue.Queue()  # Instantiate a standard queue.Queue.
    )  # End queue creation.

    # --- Sounddevice Callback (runs in a separate, dedicated audio thread managed by the sounddevice library) ---
    def audio_callback(  # Define the callback function that sounddevice will call with audio data.
        indata: np.ndarray,  # Input audio data buffer (numpy array, shape: [frames, channels]).
        frames: int,  # The number of frames in the current buffer (indata.shape[0]).
        time_info: Any,  # Timing information provided by PortAudio (usually not needed here).
        status: sd.CallbackFlags,  # Status flags indicating potential issues (e.g., buffer overflow/underflow).
    ):  # End callback function definition.
        """Callback function called by sounddevice for each new audio buffer."""  # Docstring for the callback.
        if status:  # Check if the status flags indicate any problems.
            logging.warning(  # Log any non-normal status flags as warnings.
                f"Audio Callback Status: {status}"  # Include the status flag details.
            )  # End logging warning.
        # Put a *copy* of the incoming audio data buffer onto the queue.
        # Copying is crucial because the 'indata' buffer might be reused or modified by sounddevice.
        audio_queue.put(  # Add data to the queue.
            indata.copy()  # Ensure a copy is placed in the queue.
        )  # End put call.

    # --- End Callback ---

    # Start the Enter key listener task concurrently with the recording process.
    enter_listener_task = asyncio.create_task(  # Create an asyncio task to run the Enter listener coroutine.  # Use create_task for concurrent execution.
        listen_for_enter_async()  # Specify the coroutine to run.
    )  # End task creation.

    stream = None  # Initialize stream variable to None (will hold the sounddevice InputStream).
    audio_data_np = (  # Initialize numpy array variable for concatenated audio to None.
        None  # Will hold the final complete audio recording.
    )
    try:  # Start error handling block for the audio streaming and processing part.
        # Use a context manager for the sounddevice InputStream to ensure it's properly opened and closed.
        with sd.InputStream(  # Open the audio input stream using sounddevice.
            samplerate=rate,  # Set the desired sample rate.
            channels=1,  # Set the number of audio channels (1 for mono).
            dtype="int16",  # Set the data type for audio samples (16-bit integers).
            callback=audio_callback,  # Register the callback function to receive audio data.
        ) as stream:  # Assign the opened stream object to the 'stream' variable.
            start_time = (  # Record the start time using the asyncio event loop's clock for accurate timing.
                asyncio.get_event_loop().time()  # Get the current loop time.
            )  # End start time assignment.
            # Loop continuously, checking for the stop signal or timeout.
            while (  # Condition to keep recording: the stop event must NOT be set.
                not stop_recording_event.is_set()  # Check the event state.
            ):  # End recording loop condition.
                current_time = (
                    asyncio.get_event_loop().time()
                )  # Get current time in each iteration.
                elapsed_time = current_time - start_time  # Calculate elapsed time.
                if (  # Check if the elapsed time has reached or exceeded the maximum recording duration.
                    elapsed_time >= duration  # Compare elapsed time with the limit.
                ):  # End duration check.
                    logging.debug(  # Log that the recording is stopping due to reaching the time limit.
                        f"Recording duration limit ({duration}s) reached."  # Debug message.
                    )  # End logging debug.
                    if (
                        not stop_recording_event.is_set()
                    ):  # Check again before setting to avoid race condition with Enter press.
                        stop_recording_event.set()  # Set the event to signal the loop (and potentially the Enter listener) to stop.
                    break  # Exit the recording loop explicitly.
                # Yield control back to the asyncio event loop briefly to allow other tasks (like the Enter listener) to run.
                await asyncio.sleep(  # Pause execution of this coroutine.
                    0.1  # Sleep for a short duration (100 milliseconds).
                )  # End asyncio sleep.
            # Recording loop finished (either by Enter press or timeout).

        # Ensure the Enter listener task is properly cancelled if it's still running
        # (e.g., if the recording stopped due to timeout before Enter was pressed).
        if (  # Check if the listener task has not yet completed.
            not enter_listener_task.done()  # Check the task's completion status.
        ):  # End check.
            enter_listener_task.cancel()  # Request cancellation of the task.
            try:  # Use a try/except block to gracefully handle the expected CancelledError when awaiting a cancelled task.
                await enter_listener_task  # Wait for the task cancellation to be processed.
            except (
                asyncio.CancelledError
            ):  # Catch the specific error raised upon successful cancellation.
                logging.debug(  # Log that the cancellation was handled as expected.
                    "Enter listener task successfully cancelled."  # Debug message.
                )  # End logging debug.

        logging.debug(  # Log that the recorded audio chunks are now being processed.
            "Processing recorded audio chunks from queue..."  # Debug message.
        )  # End logging debug.
        recorded_frames = (  # Initialize an empty list to collect all audio frames from the queue.
            []  # Empty list.
        )  # End list initialization.
        while (
            not audio_queue.empty()
        ):  # Loop as long as there are items in the audio queue.
            try:  # Use a try/except block to handle the unlikely case of the queue becoming empty between the check and the get.
                frame = (
                    audio_queue.get_nowait()
                )  # Retrieve an audio frame from the queue without blocking.
                recorded_frames.append(frame)  # Add the retrieved frame to the list.
            except (  # Catch the Empty exception from queue.
                queue.Empty  # Specific exception type.
            ):  # End except block.
                logging.debug(
                    "Audio queue empty during retrieval loop."
                )  # Log if this unlikely scenario happens.
                break  # Exit the loop if the queue is unexpectedly empty.

        if (
            not recorded_frames
        ):  # Check if any audio frames were actually captured and put in the queue.
            logging.warning(  # Log a warning if no audio data was recorded (e.g., immediate Enter press or issue).
                "No audio frames were captured during recording."  # Warning message.
            )  # End logging warning.
            return None  # Return None as there is no audio to transcribe.

        # Concatenate all the collected audio frames (numpy arrays) into a single large numpy array.
        audio_data_np = np.concatenate(  # Use numpy's concatenate function.
            recorded_frames,
            axis=0,  # Specify the list of arrays and the axis (0 for stacking vertically).
        )  # End concatenation.
        logging.debug(  # Log details about the final concatenated audio data.
            f"Concatenated {len(recorded_frames)} audio blocks into array of shape {audio_data_np.shape}."  # Include block count and array shape.
        )  # End logging debug.

        # --- Optional Audio Quality Check ---
        # Calculate the maximum absolute amplitude in the int16 audio data.
        # This can give a rough idea if the recording was silent or very quiet.
        max_amplitude = np.max(  # Find the maximum value.
            np.abs(audio_data_np)  # Take the absolute value of all samples first.
        )  # End max calculation.
        logging.debug(  # Log the calculated maximum amplitude.
            f"Max audio amplitude (int16 range): {max_amplitude}"  # Debug message.
        )  # End logging debug.
        amplitude_threshold = (
            500  # Define a threshold below which audio might be considered too quiet.
        )
        if (  # Check if the maximum amplitude is below the defined threshold.
            max_amplitude < amplitude_threshold  # Comparison.
        ):  # End check.
            logging.warning(  # Log a warning suggesting the audio might be too quiet.
                f"Recorded audio seems quiet (max amplitude: {max_amplitude} < {amplitude_threshold}). Check microphone levels or input."  # Warning message with values.
            )  # End logging warning.
        # --- End Quality Check ---

        # Normalize the audio data for the Whisper model.
        # Whisper typically expects audio as float32 values normalized between -1.0 and 1.0.
        audio_float32 = (  # Perform the conversion and normalization.
            audio_data_np.astype(
                np.float32
            )  # Convert the array data type from int16 to float32.
            / 32768.0  # Divide by 32768.0 (the maximum absolute value for int16) to scale the range to approximately [-1.0, 1.0].
        )  # End normalization.

        logging.info(  # Log the start of the transcription process.
            "Transcribing audio using Faster Whisper (running in thread)..."  # Indicate that the blocking Whisper call will run in a thread.
        )  # End logging info.
        whisper_options = {  # Create a dictionary to hold options for the Whisper transcribe method.
            "language": (  # Set the language option.
                language  # Use the provided language code.
                if language
                and language.lower()
                != "none"  # Check if language is provided and not explicitly "none".
                else None  # Otherwise, set to None to enable auto-detection by Whisper.
            )  # End language option setting.
            # Add other potential Whisper options here if needed, e.g., beam_size, temperature, etc.
        }  # End whisper options dictionary.
        logging.debug(  # Log the options being passed to the Whisper model.
            f"Passing options to Faster Whisper transcribe: {whisper_options}"  # Debug message.
        )  # End logging debug.

        # --- Run blocking Whisper transcription in a separate thread ---
        # Use asyncio.to_thread to run the CPU-bound/blocking transcribe method without blocking the main asyncio event loop.
        (
            segments,
            info,
        ) = await asyncio.to_thread(  # Unpack the results returned by audio_model.transcribe (a tuple of segments list and info object).  # Execute the function in a thread.
            audio_model.transcribe,  # The function to run: the transcribe method of the WhisperModel instance.
            audio_float32.flatten(),  # Pass the flattened float32 audio data as the primary argument. Flatten ensures it's 1D.
            **whisper_options,  # Pass the dictionary of options as keyword arguments.
        )  # End asyncio.to_thread call.  # End transcription thread execution and result unpacking.
        # --- End Transcription Thread ---

        # Combine the text from all transcribed segments into a single string.
        full_text = "".join(  # Use string join for efficient concatenation.
            segment.text  # Access the 'text' attribute of each segment object.
            for segment in segments  # Iterate through the list of segment objects returned by Whisper.
        )  # End join operation.
        text = (  # Remove any leading or trailing whitespace from the combined text.
            full_text.strip()  # Apply the strip() method.
        )  # End strip operation.

        # Log language detection information if available.
        if (  # Check if the 'info' object and its relevant attributes exist.
            info  # Check if info object is not None.
            and hasattr(info, "language")  # Check if it has a 'language' attribute.
            and hasattr(  # Check if it has a 'language_probability' attribute.
                info, "language_probability"  # Attribute name.
            )  # End hasattr check.
        ):  # End language info check.
            logging.debug(  # Log the detected language and its confidence probability.
                f"Whisper detected language: {info.language} (probability: {info.language_probability:.2f})"  # Format the probability to 2 decimal places.
            )  # End logging debug.

        logging.debug(  # Log the raw transcribed text before returning.
            f"Raw transcribed text: '{text}'"  # Debug message.
        )  # End logging debug.

        if (
            not text
        ):  # Check if the final transcribed text is empty (e.g., only silence was recorded).
            logging.warning(  # Log a warning if the transcription result is empty.
                "Whisper transcription resulted in empty text."  # Warning message.
            )  # End logging warning.
            return None  # Return None to indicate no meaningful speech was recognized.

        print(  # Print the recognized text to the console for immediate user feedback.
            f"Recognized: '{text}'"  # User feedback message.
        )  # End print.
        return text  # Return the successfully transcribed text string.

    except (
        sd.PortAudioError
    ) as e:  # Catch specific errors related to the PortAudio library used by sounddevice.
        logging.error(  # Log an error indicating an audio device issue.
            f"Sounddevice/PortAudio error: {e}. Check microphone configuration and availability."  # Provide specific guidance.
        )  # End logging error.
        return None  # Return None indicating failure due to audio device problems.
    except (
        Exception
    ) as e:  # Catch any other unexpected exceptions during the recording or transcription process.
        logging.error(  # Log the general error with details.
            f"Unexpected error during speech recognition: {e}\n{traceback.format_exc()}"  # Include exception message and traceback.
        )  # End logging error.
        return None  # Return None indicating a general failure.
    finally:  # Cleanup block that always executes, regardless of whether an exception occurred or not.
        # Ensure the stop event is set, especially if an error occurred mid-recording.
        # This helps ensure any waiting loops or tasks depending on the event can exit cleanly.
        if not stop_recording_event.is_set():  # Check again before setting.
            stop_recording_event.set()  # Set the event.
        # Ensure the Enter listener task is cancelled if it hasn't finished naturally or due to prior cancellation.
        if (  # Check if the listener task variable exists in the local scope and the task is not yet done.
            "enter_listener_task" in locals()  # Check if the variable was assigned.
            and enter_listener_task  # Check if it's not None.
            and not enter_listener_task.done()  # Check completion status.
        ):  # End check.
            enter_listener_task.cancel()  # Request cancellation.
            try:  # Await cancellation to handle potential errors during the cancellation process itself.
                await enter_listener_task  # Wait for cancellation.
            except asyncio.CancelledError:  # Catch the expected error.
                logging.debug(  # Log that the task was cancelled during the finally block.
                    "Enter listener task cancelled during cleanup (finally block)."  # Debug message.
                )  # End logging debug.
            except (
                Exception
            ) as final_cancel_e:  # Catch other potential errors during final cancellation await.
                logging.error(
                    f"Error awaiting cancelled enter_listener_task in finally block: {final_cancel_e}"
                )  # Log unexpected error during cleanup.


# --- LLM Generation Function (Integrated from llm_generation.py) ---
async def generate_response_async(  # Define async function for generating LLM response.
    prompt: str,  # Accept the user's input prompt string.
    llm_model: AutoModelForCausalLM,  # Accept the loaded language model object.
    tokenizer: AutoTokenizer,  # Accept the loaded tokenizer object.
    history: List[  # Accept the conversation history (list of role/content dicts).
        Dict[
            str, str
        ]  # Type hint for list elements (dictionaries with string keys and values).
    ],  # End history type hint.
) -> Optional[  # Return type hint: Optional string containing the generated response.
    str
]:  # End return type hint.
    """Generates LLM response using model.generate in a separate thread, with robust input handling."""  # Docstring explaining the function's purpose.
    if (  # Check if the model and tokenizer objects are valid.
        not llm_model
        or not tokenizer
        or not torch  # Check if any required object is None or False-like.
    ):  # End check.
        logging.error(  # Log an error if essential components are missing.
            "generate_response_async called but LLM/tokenizer/torch not available."  # Error message.
        )  # End logging error.
        return None  # Return None as generation cannot proceed.
    if (  # Validate the input prompt.
        not isinstance(prompt, str)  # Check if prompt is a string.
        or not prompt.strip()  # Check if prompt is not empty or just whitespace.
    ):  # End check.
        logging.warning(  # Log a warning if the prompt is invalid.
            "Received empty or invalid prompt for generation."  # Warning message.
        )  # End logging warning.
        return None  # Return None as there's no valid prompt to process.

    try:  # Start error handling block for text generation.
        messages = [  # Initialize the list of messages for the model input.
            {  # Start with the system prompt dictionary.
                "role": "system",  # Define the role as 'system'.
                "content": SYSTEM_PROMPT,  # Assign the predefined system prompt content.
            }  # End system prompt dictionary.
        ]  # End message list initialization.
        messages.extend(
            history
        )  # Add the conversation history turns to the messages list.
        messages.append(  # Add the current user prompt as the last message in the list.
            {  # Create a dictionary for the user turn.
                "role": "user",  # Define the role as 'user'.
                "content": prompt.strip(),  # Assign the cleaned user prompt content.
            }  # End user turn dictionary.
        )  # End append call.

        # --- Ensure pad_token_id is set for generation ---
        current_pad_token_id = (  # Get the current pad token ID from the tokenizer.
            tokenizer.pad_token_id  # Access the pad_token_id attribute.
        )  # End assignment.
        if (  # Check if pad_token_id is None but eos_token_id exists.
            current_pad_token_id is None  # Check if pad_token_id is missing.
            and tokenizer.eos_token_id
            is not None  # Check if eos_token_id is available.
        ):  # End check.
            logging.debug(  # Log that the EOS token ID will be used as the pad token ID for this generation.
                "Using eos_token_id as pad_token_id for generation as pad_token_id is None."  # Debug message.
            )  # End logging debug.
            current_pad_token_id = (  # Assign the EOS token ID to the variable used for padding.
                tokenizer.eos_token_id  # Use the eos_token_id value.
            )  # End assignment.
        elif (  # If both pad_token_id and eos_token_id are None.
            current_pad_token_id is None  # Check pad_token_id again.
        ):  # End check.
            logging.error(  # Log a critical error as generation cannot proceed without a pad token ID.
                "Cannot generate: Both pad_token_id and eos_token_id are None in the tokenizer."  # Error message.
            )  # End logging error.
            return None  # Return None indicating failure due to missing essential token IDs.

        # --- Prepare inputs (Synchronous, usually fast) ---
        # MODIFICATION START: Separate formatting and tokenization
        formatted_prompt_string = (
            None  # Initialize variable for formatted prompt string.
        )
        tokenized_inputs = None  # Initialize variable to store tokenized inputs.
        input_ids = None  # Initialize variable for input IDs tensor.
        attention_mask = None  # Initialize variable for attention mask tensor.
        try:  # Start try block specifically for formatting, tokenization and tensor preparation.
            # Step 1: Apply the chat template to get the formatted string.
            formatted_prompt_string = tokenizer.apply_chat_template(  # Call the method to apply the template.
                messages,  # Pass the list of message dictionaries.
                tokenize=False,  # Set tokenize to False to get the string.
                add_generation_prompt=True,  # Add the prompt indicating the start of the assistant's turn.
            )  # End apply_chat_template call.
            logging.debug(
                f"Formatted prompt string (first 100 chars): {formatted_prompt_string[:100]}..."
            )  # Log the beginning of the formatted string.

            # Step 2: Tokenize the formatted string using the standard tokenizer call.
            tokenized_inputs = tokenizer(  # Call the tokenizer instance directly.
                formatted_prompt_string,  # Pass the formatted string.
                return_tensors="pt",  # Request PyTorch tensors as output.
                return_attention_mask=True,  # Explicitly request the attention mask.
                # padding=True, # Consider adding padding if batching multiple sequences (not needed here for single sequence).
                # truncation=True, # Consider adding truncation if sequences might exceed model max length.
            ).to(
                llm_model.device
            )  # Move the resulting tensors to the model's device.
            logging.debug(
                f"Tokenized inputs moved to device: {llm_model.device}"
            )  # Log device placement.

            # Extract input_ids and attention_mask tensors safely.
            input_ids = tokenized_inputs.get(
                "input_ids"
            )  # Use .get() for safer access.
            attention_mask = tokenized_inputs.get(
                "attention_mask"
            )  # Use .get() for safer access.

            # Check if input_ids or attention_mask are missing after tokenization.
            if input_ids is None:  # Check if input_ids tensor is missing.
                logging.error(
                    "Tokenization failed to produce 'input_ids'."
                )  # Log error.
                return None  # Cannot proceed without input_ids.
            if attention_mask is None:  # Check if attention_mask is missing.
                logging.warning(
                    "Attention mask missing from tokenized_inputs. Creating default mask."
                )  # Log a warning.
                # Create a default attention mask of all ones with the same shape and device as input_ids.
                attention_mask = torch.ones_like(
                    input_ids, device=llm_model.device
                )  # Create mask.

        # MODIFICATION END: Separate formatting and tokenization
        except (
            IndexError
        ) as ie:  # Catch potential IndexError during tokenization or tensor access.
            logging.error(
                f"IndexError during input preparation: {ie}. Input messages: {messages}"
            )  # Log the error and the messages that caused it.
            logging.error(traceback.format_exc())  # Log the full traceback.
            return None  # Return None indicating failure.
        except (
            Exception
        ) as e:  # Catch any other unexpected errors during tokenization or tensor preparation.
            logging.error(
                f"Error during input formatting/tokenization/preparation: {e}"
            )  # Log the general error.
            logging.error(traceback.format_exc())  # Log the full traceback.
            return None  # Return None indicating failure.

        # Calculate the length of the input sequence (number of tokens).
        input_length = input_ids.shape[
            -1
        ]  # Get the size of the last dimension (sequence length).
        logging.debug(
            f"Input sequence length: {input_length} tokens."
        )  # Log input length.
        # --- End input preparation ---

        logging.info(  # Log the start of the LLM response generation process.
            f"Generating LLM response with {input_length} input tokens (running in thread)..."  # Indicate that generation will run in a separate thread and log input size.
        )  # End logging info.

        # --- Run blocking model.generate in a separate thread ---
        @torch.no_grad()  # Decorator to disable gradient calculations during inference for efficiency and memory saving.
        def _generate_sync():  # Define an inner synchronous function to encapsulate the blocking generate call.
            """Inner function to run model.generate synchronously."""  # Docstring for inner function.
            # Call the generate method of the language model.
            outputs = llm_model.generate(  # Execute the generation process.
                input_ids=input_ids,  # Pass the input token IDs tensor.
                attention_mask=attention_mask,  # Pass the attention mask tensor.
                max_new_tokens=MAX_NEW_TOKENS,  # Set the maximum number of new tokens to generate.
                temperature=0.7,  # Define the sampling temperature for generation (controls randomness). Use configured value if available.
                do_sample=True,  # Enable sampling (necessary for temperature > 0).
                pad_token_id=current_pad_token_id,  # Pass the determined pad token ID to prevent warnings/errors.
                eos_token_id=tokenizer.eos_token_id,  # Pass the EOS token ID to allow the model to stop generation naturally.
            )  # End generate call.
            return outputs  # Return the raw output tensor(s) from generate.

        outputs_tensor = await asyncio.to_thread(  # Assign the result of the threaded execution.  # Run the _generate_sync function in a separate thread managed by asyncio.
            _generate_sync  # Pass the inner synchronous function to be executed.
        )  # End asyncio.to_thread call.  # End assignment.
        # --- End Generation Thread ---

        # --- Decode response (Synchronous, usually fast) ---
        # Extract the newly generated tokens by slicing the output tensor.
        # Assumes outputs_tensor[0] contains the full sequence (input + output).
        response_tokens = outputs_tensor[  # Select the generated part of the sequence.  # Access the output tensor (usually shape [batch_size, sequence_length]).
            0  # Select the first batch item (assuming batch size 1 for typical inference).
        ][  # Start slicing the sequence dimension.
            input_length:  # Slice from the index *after* the input sequence ends, up to the end of the generated sequence.
        ]  # End slicing.

        # Decode the response tokens tensor back into a human-readable string.
        response_text = tokenizer.decode(  # Assign the decoded text.  # Call the tokenizer's decode method.
            response_tokens,  # Pass the tensor containing only the newly generated tokens.
            skip_special_tokens=True,  # Option to remove special tokens (like <eos>, <pad>) from the final output string.
        ).strip()  # Remove any leading or trailing whitespace from the decoded string.  # End decode call and strip.
        # --- End Decoding ---

        logging.info(
            f"LLM generation complete. Output length: {len(response_tokens)} tokens."
        )  # Log completion and generated token count.
        print(
            f"Generated Response: '{response_text}'"
        )  # Print the generated response directly to the console for user feedback.
        return (  # Return the response text if it's not empty, otherwise return None.
            response_text
            if response_text
            else None  # Conditional return based on whether the response is non-empty.
        )  # End return statement.

    except (  # Catch specific runtime errors like CUDA Out-of-Memory or value errors during generation.
        RuntimeError,  # Catch PyTorch runtime errors (e.g., OOM).
        ValueError,  # Catch value errors (e.g., invalid generation parameters passed).
    ) as e:  # Assign the caught exception object to 'e'.
        logging.error(  # Log the specific error encountered during generation execution.
            f"Error during text generation execution: {e}"  # Include the error message.
        )  # End logging error.
        if (  # Check if the error message indicates a CUDA Out-of-Memory error.
            "CUDA out of memory"  # String to check for in the error message.
            in str(e)  # Convert the exception to a string and perform the check.
        ):  # End OOM check.
            logging.warning(  # Log a specific warning for OOM errors with suggestions.
                "CUDA OOM Error during generation. Try reducing MAX_NEW_TOKENS, using a smaller model, or ensuring sufficient GPU memory."  # Suggestion.
            )  # End logging warning.
        logging.error(
            traceback.format_exc()
        )  # Log the full traceback for detailed debugging.
        return None  # Return None indicating generation failure.
    except (
        Exception
    ) as e:  # Catch any other unexpected exceptions during the generation process.
        logging.error(  # Log the unexpected error.
            # Use repr(e) for a more detailed representation of the exception object, including its type.
            f"Unexpected error during generation: {repr(e)}\n{traceback.format_exc()}"  # Include repr(e) and the full traceback.
        )  # End logging error.
        return None  # Return None indicating an unexpected failure.


# --- Helper function for TTS playback to run in thread ---
def _play_audio_blocking(  # Define a synchronous helper function specifically for audio playback.
    audio_data: np.ndarray,  # Accept the audio data as a numpy array (likely float32).
    sample_rate: int,  # Accept the sample rate of the audio data.
):  # End function definition.
    """Plays audio using sounddevice.play() and waits synchronously for completion."""  # Docstring explaining the function's purpose.
    try:  # Start error handling block specifically for playback.
        logging.debug(
            f"Starting playback of {len(audio_data)} samples at {sample_rate} Hz."
        )  # Log playback start details.
        sd.play(  # Call sounddevice's play function.
            audio_data, sample_rate  # Pass the audio data and its sample rate.
        )  # End play call (starts playback asynchronously in the background).
        sd.wait()  # Wait synchronously (blocks the current thread) until the playback initiated by sd.play() finishes.
        logging.debug("Playback finished.")  # Log playback completion.
    except (
        Exception
    ) as e:  # Catch any exceptions that occur during playback or waiting.
        logging.error(  # Log the error with details.
            f"Error during audio playback (_play_audio_blocking): {e}\n{traceback.format_exc()}"  # Include exception message and traceback.
        )  # End logging error.


async def speak_async(  # Define async function for text-to-speech synthesis and playback.
    text: str,
    tts_engine: TTS,  # Accept the text to speak and the initialized TTS engine object.
):  # End function definition.
    """Synthesizes speech using Coqui TTS in a thread and plays it back using sounddevice in another thread."""  # Docstring explaining the function's purpose.
    if (
        not tts_engine
    ):  # Check if the TTS engine object is valid (was initialized successfully).
        logging.error(  # Log an error if the engine is missing.
            "speak_async called but Coqui TTS engine not initialized."  # Error message.
        )  # End logging error.
        return  # Exit the function as TTS cannot proceed without the engine.
    if (
        not isinstance(text, str) or not text.strip()
    ):  # Validate the input text. Check if it's a non-empty string.
        logging.warning(  # Log a warning if the text is invalid or empty.
            "Received empty or invalid text to speak."  # Warning message.
        )  # End logging warning.
        return  # Exit the function as there's nothing to synthesize or speak.

    try:  # Start error handling block for the entire TTS synthesis and playback process.
        logging.info(  # Log the start of the speech synthesis process.
            "Synthesizing speech using Coqui TTS (running in thread)..."  # Indicate that the blocking TTS call will run in a thread.
        )  # End logging info.

        # Determine TTS arguments based on config and loaded model capabilities.
        (
            speaker_arg,  # Variable for speaker ID argument.
            language_arg,  # Variable for language code argument.
            speaker_wav_arg,  # Variable for speaker WAV path argument (for XTTS).
        ) = (  # Initialize all TTS arguments to None by default.
            None,  # Default speaker arg.
            None,  # Default language arg.
            None,  # Default speaker wav arg.
        )  # End argument initialization.

        # Check model type/capabilities to set appropriate arguments.
        tts_model_name_lower = (  # Get the lowercase version of the configured TTS model name for case-insensitive checks.
            COQUI_TTS_MODEL_NAME.lower()  # Convert to lowercase.
        )  # End lowercase conversion.
        if (  # Check if the model name suggests it's an XTTS model (which requires language and speaker_wav).
            "xtts" in tts_model_name_lower  # Check if "xtts" substring is present.
        ):  # End XTTS check.
            language_arg = (  # Set the language argument from the configuration.
                COQUI_TTS_LANGUAGE  # Assign configured language code.
            )
            speaker_wav_arg = COQUI_TTS_SPEAKER_WAV  # Set the speaker wav path argument from the configuration.  # Assign configured speaker wav path (can be None).
            logging.debug(  # Log the specific configuration being used for XTTS synthesis.
                f"Configuring TTS for XTTS model (lang: {language_arg}, speaker_wav: {speaker_wav_arg or 'using default voice if None'})"  # Include details.
            )  # End logging debug.
        elif (  # If not XTTS, check if the loaded TTS model instance indicates it's multi-speaker.
            tts_engine.is_multi_speaker  # Check the multi-speaker flag on the engine instance.
        ):  # End multi-speaker check.
            speaker_arg = (
                COQUI_TTS_SPEAKER  # Set the speaker ID argument from the configuration.
            )
            logging.debug(  # Log the speaker ID being used for the multi-speaker model.
                f"Configuring TTS for multi-speaker model (speaker: {speaker_arg})"  # Include the speaker ID.
            )  # End logging debug.

        # Check for multi-lingual capability separately, as some models might be multi-lingual but not multi-speaker,
        # or XTTS might handle language differently. Only set if not already set by XTTS logic.
        if (  # Check if the model is multi-lingual AND the language argument hasn't been set yet.
            tts_engine.is_multi_lingual and not language_arg  # Combine checks.
        ):  # End check.
            language_arg = (
                COQUI_TTS_LANGUAGE  # Set the language argument from the configuration.
            )
            logging.debug(  # Log the language code being used for the multi-lingual model.
                f"Configuring TTS for multi-lingual model (language: {language_arg})"  # Include the language code.
            )  # End logging debug.

        # --- Run blocking TTS synthesis in a separate thread ---
        # Use asyncio.to_thread to run the potentially slow/blocking tts method.
        wav_list = await asyncio.to_thread(  # Execute the function in a thread.
            tts_engine.tts,  # The function to run: the tts method of the TTS engine instance.
            text=text,  # Pass the input text string.
            speaker=speaker_arg,  # Pass the determined speaker argument (will be None if not applicable).
            language=language_arg,  # Pass the determined language argument (will be None if not applicable).
            speaker_wav=speaker_wav_arg,  # Pass the determined speaker wav path argument (will be None if not applicable).
        )  # End asyncio.to_thread call for synthesis.
        # --- End Synthesis Thread ---

        if (
            not wav_list
        ):  # Check if the TTS synthesis returned an empty result (e.g., error or empty input).
            logging.error(  # Log an error if synthesis failed to produce audio data.
                "TTS synthesis returned no audio data. Cannot speak."  # Error message.
            )  # End logging error.
            return  # Exit the function as there is nothing to play.

        # Convert the returned audio data (often a list of samples) into a numpy float32 array.
        # Playback typically requires a numpy array.
        audio_data = np.array(  # Use numpy's array constructor.
            wav_list  # Pass the result from tts_engine.tts (could be list or already an array).
        ).astype(  # Ensure the data type is float32.
            np.float32  # Target data type for sounddevice playback.
        )  # End numpy array conversion.

        # Determine the correct sample rate for playback. Try to get it from the TTS model, otherwise use a common default.
        actual_sample_rate = (  # Set a default common TTS sample rate.
            22050  # A frequent sample rate for TTS models.  # End default assignment.
        )
        try:  # Safely try to access the sample rate from the loaded model.
            if hasattr(  # Check if the TTS engine object has the 'synthesizer' attribute.
                tts_engine, "synthesizer"  # Attribute name.
            ) and hasattr(  # Check if the synthesizer object has the 'output_sample_rate' attribute.
                tts_engine.synthesizer,  # Access the synthesizer object.
                "output_sample_rate",  # Attribute name.
            ):  # End attribute check.
                actual_sample_rate = (  # Get the actual sample rate directly from the loaded TTS model's synthesizer.
                    tts_engine.synthesizer.output_sample_rate  # Access the attribute value.
                )  # End sample rate assignment.
                logging.debug(  # Log that the specific sample rate from the model is being used.
                    f"Using actual TTS model output sample rate: {actual_sample_rate} Hz."  # Include the rate.
                )  # End logging debug.
            else:  # If attributes are missing.
                logging.warning(  # Log a warning that the default rate is being assumed.
                    f"Could not determine specific TTS sample rate from model attributes. Assuming default: {actual_sample_rate} Hz."  # Include the assumed rate.
                )  # End logging warning.
        except (
            Exception
        ) as sr_e:  # Catch potential errors during sample rate retrieval.
            logging.warning(
                f"Error retrieving TTS sample rate, assuming default {actual_sample_rate} Hz: {sr_e}"
            )  # Log warning with error.

        logging.info(  # Log the start of the audio playback process.
            "Playing synthesized speech (running in thread)..."  # Indicate that playback will run in a separate thread.
        )  # End logging info.
        # --- Run blocking playback in a separate thread ---
        # Use asyncio.to_thread again to run the synchronous _play_audio_blocking helper function.
        await asyncio.to_thread(  # Execute the helper function in a thread.
            _play_audio_blocking,  # Pass the synchronous helper function.
            audio_data,  # Pass the synthesized audio data (numpy array).
            actual_sample_rate,  # Pass the determined sample rate.
        )  # End asyncio.to_thread call for playback.
        # --- End Playback Thread ---
        logging.info(  # Log that the speech playback has completed successfully.
            "Finished speaking."  # Completion message.
        )  # End logging info.

    except (  # Catch runtime errors, which might include CUDA OOM if TTS synthesis uses the GPU.
        RuntimeError  # Specific exception type.
    ) as e:  # Assign exception object to 'e'.
        logging.error(  # Log the specific runtime error encountered during synthesis or playback.
            f"Runtime error during TTS synthesis/playback: {e}"  # Include the error message.
        )  # End logging error.
        if (  # Check if the error message indicates a CUDA Out-of-Memory error.
            "CUDA out of memory"  # String to check for.
            in str(e)  # Convert exception to string and perform check.
        ):  # End OOM check.
            logging.warning(  # Log a specific warning for TTS OOM errors with suggestions.
                "CUDA OOM Error during TTS synthesis. Try using a less demanding TTS model or ensure sufficient GPU memory."  # Provide suggestions.
            )  # End logging warning.
        logging.error(traceback.format_exc())  # Log the full traceback for debugging.
    except Exception as e:  # Catch any other unexpected exceptions during the process.
        logging.error(  # Log the general error with details.
            f"Unexpected error during TTS synthesis or playback: {e}\n{traceback.format_exc()}"  # Include exception message and traceback.
        )  # End logging error.


# --- Main Asynchronous Execution ---
async def main():  # Define the main asynchronous function that orchestrates the application flow.
    """Initializes models and runs the main interaction loop."""  # Docstring for main function.
    # --- Load Models Synchronously at Startup ---
    # These are potentially slow operations, so they are done upfront before the main loop starts.
    whisper_model = load_whisper_model(
        WHISPER_MODEL_SIZE
    )  # Load the Faster Whisper model.
    llm_model, tokenizer = (  # Load the language model and its corresponding tokenizer.
        load_language_model(  # Call the loading function.
            MODEL_NAME  # Pass the configured Hugging Face model name.
        )  # End LLM/tokenizer loading.
    )  # Unpack the returned tuple.
    tts_synthesizer = init_coqui_tts()  # Initialize the Coqui TTS engine.

    # --- Check Initialization Success ---
    # Verify that all essential components were loaded/initialized successfully.
    if not all(  # Use the all() function to check if all items in the list are truthy (not None).
        [  # List of essential components.
            whisper_model,  # The loaded Whisper model object.
            llm_model,  # The loaded language model object.
            tokenizer,  # The loaded tokenizer object.
            tts_synthesizer,  # The initialized TTS engine object.
        ]  # End list of components.
    ):  # End check.
        logging.critical(  # Log a critical error if any component failed to initialize.
            "One or more essential components (Whisper, LLM, Tokenizer, TTS) failed to initialize. Cannot continue."  # Critical error message.
        )  # End logging critical.
        print(  # Inform the user via the console about the failure.
            "\nInitialization failed. Please check the logs above for specific errors. Exiting."  # User message.
        )  # End print.
        return  # Exit the main function early, preventing the loop from starting.

    print(  # Inform the user that initialization is complete and the main interaction loop is starting.
        f"\nInitialization complete. Models loaded to {DEVICE_MAP}. Starting interaction loop (Ctrl+C to exit)."  # User message including device info.
    )  # End print.
    conversation_history: List[  # Initialize an empty list to store the conversation history.  # Type hint: List containing...
        Dict[str, str]  # ...dictionaries with 'role' and 'content' keys (both strings).
    ] = (  # End type hint.  # Assign an empty list.
        []  # Empty list literal.
    )  # End history initialization.

    try:  # Start the main error handling block for the interaction loop.
        while True:  # Loop indefinitely to handle continuous user interaction cycles.
            # --- 1. Recognize Speech Asynchronously ---
            logging.info(
                "Starting new interaction cycle: Listening..."
            )  # Log start of cycle.
            user_command = await recognize_speech_async(  # Await the result of the asynchronous speech recognition function.
                whisper_model,  # Pass the loaded Whisper model instance.
                RECORD_SECONDS,  # Pass the configured maximum recording duration.
                SAMPLE_RATE,  # Pass the configured audio sample rate.
                WHISPER_LANGUAGE,  # Pass the configured target language for transcription (or None).
            )  # End await recognize_speech_async.

            # --- 2. Process Recognized Command ---
            if (
                user_command
            ):  # Check if speech recognition was successful and returned non-empty text.  # Check if user_command is truthy (not None and not an empty string).  # End check.
                logging.info(
                    f"User command recognized: '{user_command}'"
                )  # Log recognized command.
                current_user_turn = {  # Create a dictionary representing the current user's turn for history.
                    "role": "user",  # Set the role to 'user'.
                    "content": user_command,  # Set the content to the recognized command string.
                }  # End user turn dictionary.

                # Prepare the history context for the LLM.
                # Combine the persistent history with the current user turn.
                history_for_llm = (  # Create a temporary list for LLM input.
                    conversation_history  # Start with the existing history.
                    + [current_user_turn]  # Append the current user turn dictionary.
                )  # End list combination.

                # Calculate the maximum number of *items* (dicts) to keep in the history sent to the LLM.
                # MAX_HISTORY_LENGTH refers to turns (user+assistant pairs).
                max_history_items = (  # Calculate the limit in terms of individual messages.
                    MAX_HISTORY_LENGTH
                    * 2  # Multiply max turns by 2 (user message + assistant message).
                )  # End calculation.

                # Trim the history *sent to the LLM* if it exceeds the calculated item limit.
                if (  # Check if the length of the temporary history list exceeds the limit.
                    len(history_for_llm)
                    > max_history_items  # Compare length with the limit.
                ):  # End length check.
                    # Slice the list to keep only the most recent 'max_history_items' elements.
                    history_for_llm = history_for_llm[  # Reassign the sliced list.
                        -max_history_items:  # Negative index slices from the end.
                    ]  # End slicing.
                    logging.debug(  # Log that the history context for the LLM was trimmed.
                        f"History context for LLM input trimmed to the last {len(history_for_llm)} messages ({MAX_HISTORY_LENGTH} turns)."  # Debug message.
                    )  # End logging debug.

                # --- 3. Generate Response Asynchronously ---
                logging.info("Generating AI response...")  # Log generation start.
                ai_response = await generate_response_async(  # Await the result of the asynchronous LLM response generation function.
                    user_command,  # Pass the recognized user command as the prompt.
                    llm_model,  # Pass the loaded language model instance.
                    tokenizer,  # Pass the loaded tokenizer instance.
                    history_for_llm,  # Pass the prepared (potentially trimmed) history list for context.
                )  # End await generate_response_async.

                # --- 4. Process and Speak Response ---
                if (
                    ai_response
                ):  # Check if the LLM successfully generated a non-empty response.
                    logging.info(
                        f"AI response generated: '{ai_response[:100]}...'"
                    )  # Log beginning of response.
                    # Add the user turn AND the successful AI response to the persistent conversation history.
                    conversation_history.append(  # Add the user's turn dictionary to the main history.
                        current_user_turn  # The dictionary created earlier.
                    )  # End append call.
                    conversation_history.append(  # Add the assistant's turn dictionary to the main history.
                        {  # Create the assistant turn dictionary.
                            "role": "assistant",  # Role is assistant.
                            "content": ai_response,  # Content is the generated response string.
                        }  # End assistant turn dictionary.
                    )  # End append call.

                    # Trim the main persistent history list if it grows too large, keeping slightly more than the LLM context if desired.
                    # Using the same limit as LLM context here for simplicity.
                    if (  # Check if the persistent history now exceeds the item limit.
                        len(conversation_history)
                        > max_history_items  # Compare length with limit.
                    ):  # End check.
                        # Trim the persistent history list from the beginning.
                        conversation_history = conversation_history[  # Reassign the sliced list.
                            -max_history_items:  # Keep the last 'max_history_items'.
                        ]  # End slicing.
                        logging.debug(  # Log that the main persistent history list was trimmed.
                            f"Main conversation history trimmed to {len(conversation_history)} messages ({MAX_HISTORY_LENGTH} turns)."  # Debug message.
                        )  # End logging debug.

                    # --- 5. Speak Response Asynchronously ---
                    logging.info("Speaking AI response...")  # Log speaking start.
                    await speak_async(  # Await the completion of the asynchronous text-to-speech function.
                        ai_response,  # Pass the generated AI response string.
                        tts_synthesizer,  # Pass the initialized TTS engine instance.
                    )  # End await speak_async.
                else:  # Handle cases where LLM response generation failed or returned None/empty.
                    logging.warning(  # Log a warning that no valid response was generated.
                        "LLM response generation failed or returned empty."  # Warning message.
                    )  # End logging warning.
                    print(  # Inform the user via the console.
                        "Sorry, I couldn't generate a response for that."  # User feedback message.
                    )  # End print.
                    # Optionally, speak a fallback message.
                    await speak_async(  # Speak a predefined fallback message.
                        "Sorry, I couldn't generate a response for that.",  # Fallback text.
                        tts_synthesizer,  # Pass the TTS engine.
                    )  # End await speak_async.
            else:  # Handle cases where speech recognition failed or returned None/empty.
                logging.warning(  # Log a warning that no command was recognized.
                    "Speech recognition failed or returned empty text."  # Warning message.
                )  # End logging warning.
                print(  # Inform the user via the console.
                    "I didn't catch that. Could you please repeat?"  # User feedback message.
                )  # End print.
                # Speak a fallback message asking the user to repeat.
                await speak_async(  # Speak the fallback message.
                    "I didn't catch that. Could you please repeat?",  # Fallback text.
                    tts_synthesizer,  # Pass the TTS engine.
                )  # End await speak_async.

            # Small sleep at the end of the loop to prevent tight looping in case of continuous errors
            # and to ensure other potential asyncio tasks get a chance to run.
            await asyncio.sleep(0.1)  # Yield control back to the event loop briefly.

    except (  # Catch asyncio.CancelledError, typically raised when the main task is cancelled (e.g., by Ctrl+C).
        asyncio.CancelledError  # Specific exception type.
    ):  # End except block.
        logging.info(
            "Main interaction loop cancelled."
        )  # Log that the loop was intentionally cancelled.
    except (  # Catch KeyboardInterrupt explicitly (Ctrl+C).
        KeyboardInterrupt  # Specific exception type.
    ):  # End except block.
        print(  # Inform the user that the program is exiting due to Ctrl+C.
            "\nCtrl+C detected. Exiting program gracefully."  # User message.
        )  # End print.
        logging.info("KeyboardInterrupt received, shutting down.")  # Log the interrupt.
    except (  # Catch any other unexpected exceptions that might occur within the main loop.
        Exception  # Catch the base Exception class to be safe.
    ) as e:  # Assign the exception object to 'e'.
        logging.error(  # Log the unexpected error with details.
            f"Unexpected error in main interaction loop: {e}\n{traceback.format_exc()}"  # Include exception message and traceback.
        )  # End logging error.
        print(
            f"\nAn unexpected error occurred: {e}. Check logs. Exiting."
        )  # Inform user about unexpected error.
    finally:  # Cleanup block that always executes when the loop terminates (normally or due to exception/cancellation).
        print("Program finished.")  # Indicate that the program is terminating.
        logging.info("Main function finished execution.")  # Log program finish.
        # Optional: Add any explicit resource cleanup here if needed (though context managers handle most).
        # For example, explicitly closing audio streams if not using context managers.


if (  # Standard Python entry point check: ensures the code runs only when the script is executed directly.
    __name__ == "__main__"  # Check if the script's name is "__main__".
):  # End check.
    try:  # Start top-level error handling for the asyncio application startup itself.
        asyncio.run(
            main()
        )  # Run the main asynchronous function using asyncio.run(). This starts the event loop.
    except (  # Catch Ctrl+C pressed during the initial asyncio.run() setup or final shutdown phases.
        KeyboardInterrupt  # Specific exception type.
    ):  # End except block.
        print(  # Inform the user about the exit reason.
            "\nExiting due to KeyboardInterrupt during startup or shutdown phase."  # User message.
        )  # End print.
    except (  # Catch any other exceptions that might occur outside the main() function's try/except block (e.g., during asyncio setup).
        Exception  # Catch base Exception class.
    ) as e:  # Assign exception object to 'e'.
        logging.critical(  # Log critical errors occurring at the top level.
            f"Critical error during asyncio execution setup or final cleanup: {e}\n{traceback.format_exc()}"  # Include exception details and traceback.
        )  # End logging critical.
        print(
            f"\nA critical error occurred during startup/shutdown: {e}. Check logs."
        )  # Inform user about critical error.
