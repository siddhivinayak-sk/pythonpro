# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import os
import azure.cognitiveservices.speech as speechsdk

# Set up the speech translation config using environment variables or placeholders
speech_key = os.environ.get('AZURE_SPEECH_KEY', '<your-api-key>')
speech_region = os.environ.get('AZURE_SPEECH_REGION', 'eastus')

# Source and target languages
from_language = 'en-US'
to_languages = ['de', 'fr', 'it']

# ===== Configuration Flags =====
USE_SINGLE_SHOT = False      # Set to True for single-shot recognition, False for continuous (default)
ENABLE_SYNTHESIS = True      # Set to True to enable speech synthesis of translations
USE_LIVE_INTERPRETER = False # Set to True for Live Interpreter mode (auto-detect language, personal voice)


def translate_speech_live_interpreter():
    """
    Live Interpreter mode - Real-time speech-to-speech translation with automatic language detection.
    
    Features:
    - No need to specify source language (auto-detection)
    - Personal voice synthesis (preserves speaker's style and tone)
    - Low latency speech-to-speech translation
    - Supports language switching within the same session
    
    Note: Requires personal voice access. Apply at https://aka.ms/customneural
    """
    # Live Interpreter requires the v2 endpoint
    v2_endpoint = f"wss://{speech_region}.stt.speech.microsoft.com/speech/universal/v2"
    
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        endpoint=v2_endpoint,
        subscription=speech_key
    )
    
    # Add target language (Live Interpreter typically uses single target)
    target_language = to_languages[0] if to_languages else 'de'
    translation_config.add_target_language(target_language)
    
    # Enable personal voice for natural-sounding translation
    translation_config.voice_name = "personal-voice"
    
    # Use auto-detect without specifying source language candidates
    auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=[]  # Empty for open range detection
    )
    
    # Create audio config for microphone input
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create translation recognizer with auto-detect
    translation_recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config,
        auto_detect_source_language_config=auto_detect_config,
        audio_config=audio_config
    )
    
    done = False
    audio_file_index = [0]  # Use list for nonlocal modification
    
    def recognizing_cb(evt):
        """Callback for intermediate recognition results with detected language."""
        detected_lang = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult, "")
        print(f'RECOGNIZING in "{detected_lang}": {evt.result.text}')
        for language, translation in evt.result.translations.items():
            print(f'    TRANSLATING into "{language}": {translation}')
    
    def recognized_cb(evt):
        """Callback for final recognition results."""
        if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
            detected_lang = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult, "")
            print(f'\nRECOGNIZED in "{detected_lang}": {evt.result.text}')
            for language, translation in evt.result.translations.items():
                print(f'    TRANSLATED into "{language}": {translation}')
        elif evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f'RECOGNIZED: {evt.result.text}')
            print('    Speech not translated.')
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('NOMATCH: Speech could not be recognized.')
    
    def synthesizing_cb(evt):
        """Callback for synthesized audio (personal voice)."""
        audio = evt.result.audio
        size = len(audio)
        print(f'Audio synthesized: {size:,} byte(s) {"(Complete)" if size == 0 else ""}')
        
        if size > 0:
            audio_file_index[0] += 1
            with open(f'live_interpreter_output_{audio_file_index[0]}.wav', 'wb') as f:
                f.write(audio)
    
    def canceled_cb(evt):
        """Callback for cancellation events."""
        print(f'CANCELED: Reason={evt.cancellation_details.reason}')
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f'CANCELED: ErrorDetails={evt.cancellation_details.error_details}')
        nonlocal done
        done = True
    
    def session_stopped_cb(evt):
        """Callback for session stopped events."""
        print('\nSession stopped.')
        nonlocal done
        done = True
    
    # Connect callbacks to events
    translation_recognizer.recognizing.connect(recognizing_cb)
    translation_recognizer.recognized.connect(recognized_cb)
    translation_recognizer.synthesizing.connect(synthesizing_cb)
    translation_recognizer.canceled.connect(canceled_cb)
    translation_recognizer.session_stopped.connect(session_stopped_cb)
    
    # Start Live Interpreter
    print("=== Live Interpreter Mode ===")
    print(f"Translating to: {target_language}")
    print("Speak in any language - it will be auto-detected and translated.")
    print("Press Ctrl+C to stop...\n")
    
    translation_recognizer.start_continuous_recognition()
    
    try:
        while not done:
            pass
    except KeyboardInterrupt:
        print("\nStopping...")
    
    translation_recognizer.stop_continuous_recognition()


def translate_speech_continuous():
    """
    Event-based continuous speech translation with recognizing/recognized/synthesizing events.
    This is the recommended approach for real-time translation scenarios.
    """
    # Create translation config
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=speech_key, 
        region=speech_region
    )
    
    # Set source language
    translation_config.speech_recognition_language = from_language
    
    # Add target languages
    for lang in to_languages:
        translation_config.add_target_language(lang)
    
    # Optional: Set voice for synthesis (must match one of the target languages)
    # Note: Event-based synthesis works only with a single translation target
    if ENABLE_SYNTHESIS:
        translation_config.voice_name = "de-DE-ConradNeural"
    
    # Create audio config for microphone input
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create translation recognizer
    translation_recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, 
        audio_config=audio_config
    )
    
    # Flag to track when to stop
    done = False
    
    def recognizing_cb(evt):
        """Callback for intermediate recognition results."""
        print(f'RECOGNIZING in "{from_language}": {evt.result.text}')
        for language, translation in evt.result.translations.items():
            print(f'    TRANSLATING into "{language}": {translation}')
    
    def recognized_cb(evt):
        """Callback for final recognition results."""
        if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
            print(f'\nRECOGNIZED in "{from_language}": {evt.result.text}')
            for language, translation in evt.result.translations.items():
                print(f'    TRANSLATED into "{language}": {translation}')
        elif evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f'RECOGNIZED: {evt.result.text}')
            print('    Speech not translated.')
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('NOMATCH: Speech could not be recognized.')
    
    def synthesizing_cb(evt):
        """Callback for synthesized audio."""
        audio = evt.result.audio
        size = len(audio)
        print(f'Audio synthesized: {size:,} byte(s) {"(Complete)" if size == 0 else ""}')
        
        if size > 0:
            # Save synthesized audio to file
            with open('translation_output.wav', 'wb') as f:
                f.write(audio)
    
    def canceled_cb(evt):
        """Callback for cancellation events."""
        print(f'CANCELED: Reason={evt.cancellation_details.reason}')
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f'CANCELED: ErrorDetails={evt.cancellation_details.error_details}')
            print('CANCELED: Did you set the speech resource key and region values?')
        nonlocal done
        done = True
    
    def session_stopped_cb(evt):
        """Callback for session stopped events."""
        print('\nSession stopped.')
        nonlocal done
        done = True
    
    # Connect callbacks to events
    translation_recognizer.recognizing.connect(recognizing_cb)
    translation_recognizer.recognized.connect(recognized_cb)
    translation_recognizer.synthesizing.connect(synthesizing_cb)
    translation_recognizer.canceled.connect(canceled_cb)
    translation_recognizer.session_stopped.connect(session_stopped_cb)
    
    # Start continuous recognition
    print(f"Say something in '{from_language}' and we'll translate into {to_languages}.")
    print("Press Ctrl+C to stop...\n")
    
    translation_recognizer.start_continuous_recognition()
    
    try:
        # Wait for completion
        while not done:
            pass
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Stop recognition
    translation_recognizer.stop_continuous_recognition()


def translate_speech_single_shot():
    """
    Single-shot speech translation - recognizes a single utterance.
    Use this for simple scenarios where you only need to translate one phrase.
    """
    # Create translation config
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=speech_key, 
        region=speech_region
    )
    
    # Set source language
    translation_config.speech_recognition_language = from_language
    
    # Add target languages
    for lang in to_languages:
        translation_config.add_target_language(lang)
    
    # Create audio config for microphone input
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create translation recognizer
    translation_recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, 
        audio_config=audio_config
    )
    
    print(f"Say something in '{from_language}' and we'll translate into {to_languages}...")
    
    # Perform single-shot translation
    result = translation_recognizer.recognize_once()
    
    # Process the result
    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        print(f'RECOGNIZED "{from_language}": {result.text}')
        for language in to_languages:
            translation = result.translations.get(language, '')
            print(f'    TRANSLATED into "{language}": {translation}')
    
    elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f'Recognized: "{result.text}"')
        print('    Speech not translated.')
    
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print(f'No speech could be recognized: {result.no_match_details}')
    
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print(f'Speech Recognition canceled: {cancellation.reason}')
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f'Error details: {cancellation.error_details}')


if __name__ == "__main__":
    if USE_LIVE_INTERPRETER:
        translate_speech_live_interpreter()
    elif USE_SINGLE_SHOT:
        translate_speech_single_shot()
    else:
        translate_speech_continuous()
