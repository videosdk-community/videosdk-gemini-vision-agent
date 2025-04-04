from collections import deque  # For storing latest screen frames with fixed size
from queue import Queue
import traceback
import os
from videosdk import MeetingConfig, VideoSDK, Participant, Stream  # Video SDK for meeting functionality
from rtc.videosdk.meeting_handler import MeetingHandler  # Event handlers for meeting events
from rtc.videosdk.participant_handler import ParticipantHandler  # Event handlers for participant events
from agent.audio_stream_track import CustomAudioStreamTrack  # Custom audio track for the AI agent
from intelligence.openai.openai_intelligence import OpenAIIntelligence  # OpenAI integration
from utils.struct.openai import InputAudioTranscription  # Structure for audio transcription

import google.generativeai as genai  # Google's Gemini API for image analysis

import librosa  # Audio processing library
import numpy as np  # Numerical operations
import asyncio  # Asynchronous programming
from PIL import Image  # Image processing
import dotenv  # Environment variable loading

# Load environment variables from .env file
dotenv.load_dotenv()

# Get API keys from environment variables
openai_api_key=os.getenv("OPENAI_API_KEY")  # For audio transcription and conversation
gemini_api_key=os.getenv("GEMINI_API_KEY")  # For screen analysis


class AIAgent:
    """
    An AI agent that can join video meetings, process audio from participants,
    and analyze shared screens using OpenAI and Gemini APIs.
    """
    def __init__(self, meeting_id: str, authToken: str, name: str):
        """
        Initialize the AI agent with meeting details.
        
        Args:
            meeting_id: ID of the meeting to join
            authToken: Authentication token for the video SDK
            name: Display name of the AI agent in the meeting
        """
        
        # Create directory for storing screenshots
        os.makedirs("test_img", exist_ok=True)
        
        # Get the current event loop for async operations
        self.loop = asyncio.get_event_loop()
        
        # Create custom audio track for the agent to speak
        self.audio_track = CustomAudioStreamTrack(
            loop=self.loop,
            handle_interruption=True  # Allow interruptions in speech
        )
        
        # Configure meeting settings
        self.meeting_config = MeetingConfig(
            name=name,
            meeting_id=meeting_id,
            token=authToken,
            mic_enabled=True,  # Enable microphone for the agent
            webcam_enabled=False,  # No video feed for the agent
            custom_microphone_audio_track=self.audio_track,  # Use custom audio track
        )
        
        # Track tasks for each audio and screenshare stream
        self.audio_listener_tasks = {}
        self.screenshare_listener_tasks = {}
        
        # Initialize the meeting agent
        self.agent = VideoSDK.init_meeting(**self.meeting_config)
        
        # Add event listeners for meeting events
        self.agent.add_event_listener(
            MeetingHandler(
                on_meeting_joined=self.on_meeting_joined,
                on_meeting_left=self.on_meeting_left,
                on_participant_joined=self.on_participant_joined,
                on_participant_left=self.on_participant_left,
            ))
        
        # Initialize Gemini vision model for screen analysis
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.vision_model = None
            print("GEMINI_API_KEY not set. Screen share analysis will be disabled.")
        
        # Define tool for OpenAI to analyze screen
        screen_tool = {
            "type": "function",
            "name": "analyze_screen",
            "description": "Analyze current screen content when user asks for help with visible elements",
            "parameters": {"type": "object", "properties": {}}
        }

        # Initialize OpenAI for real-time audio transcription and processing
        self.intelligence = OpenAIIntelligence(
            loop=self.loop,
            api_key=openai_api_key,
            base_url="api.openai.com",  # API endpoint
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),  # Whisper model for transcription
            tools=[screen_tool],  # Tools available to the AI
            audio_track=self.audio_track,  # Audio track for output
            handle_function_call=self.handle_function_call,  # Handler for tool calls
        )
        
        # Queue to store the most recent screen frame (only keeps the latest one)
        self.frame_queue = deque(maxlen=1)
        self.latest_frame = None  # Store the latest frame for immediate access
        
    async def handle_function_call(self, function_call):
        """
        Handle function calls from OpenAI, particularly for screen analysis.
        
        Args:
            function_call: The function call object from OpenAI
            
        Returns:
            Analysis result or error message
        """
        if function_call.name == "analyze_screen":
            if not self.latest_frame:
                return "No screen content available"
            
            # Convert frame to image
            image_data = self.latest_frame.to_ndarray()
            image = Image.fromarray(image_data)
            
            try:
                # Request analysis from Gemini
                response = await self.loop.run_in_executor(
                    None,  # Use default executor
                    lambda: self.vision_model.generate_content([
                        "Analyze this screen to help user. Focus on relevant UI elements, text, code, and context.",
                        image
                    ])
                )
                return response.text
            except Exception as e:
                return f"Analysis error: {str(e)}"
        return "Unknown command"
    

    async def add_audio_listener(self, stream: Stream, peer_name: str):
        """
        Process audio from a participant and send it to OpenAI for transcription.
        
        Args:
            stream: The audio stream
            peer_name: Name of the participant
        """
        print("Participant stream enabled", peer_name)
        while True:
            try:
                await asyncio.sleep(0.01)  # Small delay to prevent CPU hogging

                # Get audio frame
                frame = await stream.track.recv()      
                audio_data = frame.to_ndarray()[0]
                
                # Convert to float for processing
                audio_data_float = (
                    audio_data.astype(np.float32) / np.iinfo(np.int16).max
                )
                
                # Convert to mono and resample to 16kHz (required by Whisper)
                audio_mono = librosa.to_mono(audio_data_float.T)
                audio_resampled = librosa.resample(
                    audio_mono, orig_sr=48000, target_sr=16000
                )
                
                # Convert back to PCM format for OpenAI
                pcm_frame = (
                    (audio_resampled * np.iinfo(np.int16).max)
                    .astype(np.int16)
                    .tobytes()
                )
                
                # Send to OpenAI for processing
                await self.intelligence.send_audio_data(pcm_frame)

            except Exception as e:
                print("Audio processing error:", e)
                break

    async def add_screenshare_listener(self, stream: Stream, peer_name: str):
        """
        Store the latest frame from a screen share stream.
        
        Args:
            stream: The screen share stream
            peer_name: Name of the participant sharing the screen
        """
        print("Participant screenshare enabled", peer_name)
        while True:
            try:                
                frame = await stream.track.recv()
                self.latest_frame = frame  # Update latest frame
            except Exception as e:
                traceback.print_exc()
                print("Screenshare processing error:", e)
                break
            

    async def get_screen(self):
        """
        Periodically capture and save screenshots, then analyze them with Gemini.
        """
        try:
            counter = 1
            while True:
                await asyncio.sleep(1)  # Check every second
                if len(self.frame_queue) > 0:
                    frame = self.frame_queue[0]
                    image_data = frame.to_ndarray()
                    image = Image.fromarray(image_data)
                    
                    # Save the image
                    image_path=f"test_img/screenshare_{counter}.png"
                    image.save(image_path)
                    print(f"Saved screenshot {counter}")
                    
                    # Analyze with Gemini
                    if self.vision_model:
                        try:
                            response = await self.loop.run_in_executor(
                                None,  # Uses default executor
                                lambda: self.vision_model.generate_content(
                                    ["Analyze this shared screen content. Describe any important visual information, text, code, or diagrams.", image]
                                )
                            )
                            print("Gemini Analysis:", response.text)
                        except Exception as e:
                            print("Gemini API Error:", str(e))
                    
                    counter += 1
                else:
                    print("No frame available")
        except Exception as e:
            traceback.print_exc()
            print("Error while getting frame from queue:", e)
        
    def on_meeting_joined(self, data):
        """
        Handler for when the agent joins a meeting.
        Connects the OpenAI intelligence module.
        """
        print("Meeting Joined!")
        asyncio.create_task(self.intelligence.connect())
    
    def on_meeting_left(self, data):
        """Handler for when the agent leaves a meeting."""
        print(f"Meeting Left")
        
    def on_participant_joined(self, participant: Participant):
        """
        Handler for when a participant joins the meeting.
        Sets up listeners for their audio and screen share streams.
        
        Args:
            participant: The participant who joined
        """
        peer_name = participant.display_name
        print("Participant joined:", peer_name)
        
        # Set instructions for the AI assistant
        intelligence_instructions = """
        You are an AI meeting assistant. Follow these rules:
        1. Use analyze_screen tool when user asks about:
        - Visible UI elements
        - On-screen content
        - Application help
        - Workflow guidance
        2. Keep responses under 2 sentences
        3. Always acknowledge requests first
        """

        # Update OpenAI with instructions
        asyncio.create_task(self.intelligence.update_session_instructions(intelligence_instructions))

        def on_stream_enabled(stream: Stream):
            """
            Handler for when a participant enables a stream (audio or screen share).
            
            Args:
                stream: The enabled stream
            """
            print(f"stream kind : {stream.kind}")
            if stream.kind == "audio":
                # Start processing audio
                self.audio_listener_tasks[stream.id] = self.loop.create_task(
                    self.add_audio_listener(stream, peer_name)
                )
            elif stream.kind == "share":
                # Start processing screen share
                self.screenshare_listener_tasks[stream.id] = self.loop.create_task(
                    self.add_screenshare_listener(stream, peer_name)
                )
                self.screenshare_listener_tasks[f"${stream.id}-queue"] = self.loop.create_task(
                    self.get_screen()
                )

        def on_stream_disabled(stream: Stream):
            """
            Handler for when a participant disables a stream.
            Cancels the corresponding task.
            
            Args:
                stream: The disabled stream
            """
            print("Participant stream disabled")
            if stream.kind == "audio":
                audio_task = self.audio_listener_tasks[stream.id]
                if audio_task is not None:
                    audio_task.cancel()
            elif stream.kind == "share":
                print("Participant screenshare disabled", peer_name)
                screenshare_task = self.screenshare_listener_tasks.get(stream.id)
                screen_get_task = self.screenshare_listener_tasks.get(f"${stream.id}-queue")
                if screenshare_task:
                    screenshare_task.cancel()
                    del self.screenshare_listener_tasks[stream.id]
                if screen_get_task:
                    screen_get_task.cancel()
                    del self.screenshare_listener_tasks[f"${stream.id}-queue"]
                    
        # Add event listeners for the participant
        participant.add_event_listener(
            ParticipantHandler(
                participant_id=participant.id,
                on_stream_enabled=on_stream_enabled,
                on_stream_disabled=on_stream_disabled
            )
        )

    def on_participant_left(self, participant: Participant):
        """Handler for when a participant leaves the meeting."""
        print("Participant left:", participant.display_name)
          
    async def join(self):
        """Join the meeting asynchronously."""
        await self.agent.async_join()
    
    def leave(self):
        """Leave the meeting."""
        self.agent.leave()

    async def cleanup(self):
        """
        Cleanup resources when the agent is destroyed.
        Cancels all tasks and leaves the meeting.
        """
        # Cancel all running tasks
        for task in self.audio_listener_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self.screenshare_listener_tasks.values():
            if task and not task.done():
                task.cancel()
        
        # Clear the queues
        self.frame_queue.clear()
        
        # Leave the meeting
        self.leave()