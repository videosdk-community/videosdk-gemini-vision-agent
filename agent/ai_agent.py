from collections import deque
from queue import Queue
import traceback
import os
from videosdk import MeetingConfig, VideoSDK, Participant, Stream
from rtc.videosdk.meeting_handler import MeetingHandler
from rtc.videosdk.participant_handler import ParticipantHandler
from agent.audio_stream_track import CustomAudioStreamTrack
from intelligence.openai.openai_intelligence import OpenAIIntelligence
from utils.struct.openai import InputAudioTranscription

import google.generativeai as genai  # Changed from Google Cloud Vision

import librosa
import numpy as np
import asyncio
from PIL import Image
import dotenv

dotenv.load_dotenv()

openai_api_key=os.getenv("OPENAI_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")


class AIAgent:
    def __init__(self, meeting_id: str, authToken: str, name: str):
        
        # Create test_img directory if it doesn't exist
        os.makedirs("test_img", exist_ok=True)
        
        self.loop = asyncio.get_event_loop()
        self.audio_track = CustomAudioStreamTrack(
            loop=self.loop,
            handle_interruption=True
        )
        self.meeting_config = MeetingConfig(
            name=name,
            meeting_id=meeting_id,
            token=authToken,
            mic_enabled=True,
            webcam_enabled=False,
            custom_microphone_audio_track=self.audio_track,
        )
        self.audio_listener_tasks = {}
        self.screenshare_listener_tasks = {}
        self.agent = VideoSDK.init_meeting(**self.meeting_config)
        self.agent.add_event_listener(
            MeetingHandler(
                on_meeting_joined=self.on_meeting_joined,
                on_meeting_left=self.on_meeting_left,
                on_participant_joined=self.on_participant_joined,
                on_participant_left=self.on_participant_left,
            ))
        
        # Initialize Gemini
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.vision_model = None
            print("GEMINI_API_KEY not set. Screen share analysis will be disabled.")
        
        # tools for OpenAI
        screen_tool = {
            "type": "function",
            "name": "analyze_screen",
            "description": "Analyze current screen content when user asks for help with visible elements",
            "parameters": {"type": "object", "properties": {}}
        }

        # openai realtime api
        self.intelligence = OpenAIIntelligence(
            loop=self.loop,
            api_key=openai_api_key,
            base_url="api.openai.com",  # Verify correct API endpoint
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
            tools=[screen_tool],
            audio_track=self.audio_track,
            handle_function_call=self.handle_function_call,
        )
        
        self.frame_queue: deque = deque(maxlen=1)
        
        
    async def handle_function_call(self, function_call):
        if function_call.name == "analyze_screen":
            if not self.latest_frame:
                return "No screen content available"
            
            image_data = self.latest_frame.to_ndarray()
            image = Image.fromarray(image_data)
            
            try:
                if function_call.name == "analyze_screen":
                    if not self.latest_frame:
                        return "No screen content available"
                response = await self.loop.run_in_executor(
                    None,
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
        print("Participant stream enabled", peer_name)
        while True:
            try:
                await asyncio.sleep(0.01)

                frame = await stream.track.recv()      
                audio_data = frame.to_ndarray()[0]
                audio_data_float = (
                    audio_data.astype(np.float32) / np.iinfo(np.int16).max
                )
                audio_mono = librosa.to_mono(audio_data_float.T)
                audio_resampled = librosa.resample(
                    audio_mono, orig_sr=48000, target_sr=16000
                )
                pcm_frame = (
                    (audio_resampled * np.iinfo(np.int16).max)
                    .astype(np.int16)
                    .tobytes()
                )
                
                await self.intelligence.send_audio_data(pcm_frame)

            except Exception as e:
                print("Audio processing error:", e)
                break

    async def add_screenshare_listener(self, stream: Stream, peer_name: str):
        """Store latest frame only"""
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
        """Capture and save screenshots."""
        try:
            counter = 1
            while True:
                await asyncio.sleep(1)
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
        print("Meeting Joined!")
        asyncio.create_task(self.intelligence.connect())
    
    def on_meeting_left(self, data):
        print(f"Meeting Left")
        
    def on_participant_joined(self, participant: Participant):
        peer_name = participant.display_name
        print("Participant joined:", peer_name)
        
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

        asyncio.create_task(self.intelligence.update_session_instructions(intelligence_instructions))

        def on_stream_enabled(stream: Stream):
            print(f"stream kind : {stream.kind}")
            if stream.kind == "audio":
                self.audio_listener_tasks[stream.id] = self.loop.create_task(
                    self.add_audio_listener(stream, peer_name)
                )
            elif stream.kind == "share":
                self.screenshare_listener_tasks[stream.id] = self.loop.create_task(
                    self.add_screenshare_listener(stream, peer_name)
                )
                self.screenshare_listener_tasks[f"${stream.id}-queue"] = self.loop.create_task(
                    self.get_screen()
                )

        def on_stream_disabled(stream: Stream):
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
                    
        participant.add_event_listener(
            ParticipantHandler(
                participant_id=participant.id,
                on_stream_enabled=on_stream_enabled,
                on_stream_disabled=on_stream_disabled
            )
        )

    def on_participant_left(self, participant: Participant):
        print("Participant left:", participant.display_name)
          
    async def join(self):
        await self.agent.async_join()
    
    def leave(self):
        self.agent.leave()

    async def cleanup(self):
        """Cleanup resources when the agent is destroyed."""
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