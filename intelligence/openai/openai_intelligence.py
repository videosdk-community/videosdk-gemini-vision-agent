import base64  # For encoding/decoding binary audio data
import traceback  # For detailed exception information
from typing import Dict, List, Union, Callable, Optional  # Type hints
from utils.struct.openai import (
    # Import various OpenAI-related data structures and utilities
    AudioFormats,  # Enum for audio formats (PCM16, etc.)
    ClientToServerMessage,  # Base class for messages sent to OpenAI
    EventType,  # Enum of event types received from OpenAI
    FunctionCallOutputItemParam,  # Structure for function call results
    InputAudioBufferAppend,  # Message type to send audio data
    InputAudioTranscription,  # Configuration for audio transcription
    ItemCreate,  # Message type to create an item (like function results)
    ItemParam,  # Base parameter type for items
    ResponseCreate,  # Message type to create a response
    ResponseCreateParams,  # Parameters for response creation
    ResponseFunctionCallArgumentsDone,  # Event when function call is ready
    ServerVADUpdateParams,  # Voice Activity Detection parameters
    SessionUpdate,  # Message type to update session config
    SessionUpdateParams,  # Parameters for session updates
    ToolResult,  # Structure for tool call results
    Voices,  # Enum of available TTS voices
    generate_event_id,  # Utility to generate unique event IDs
    to_json,  # Convert message to JSON
)
import json  # For parsing JSON responses

from asyncio.log import logger  # Logging
from asyncio import AbstractEventLoop  # Event loop for async operations
import aiohttp  # HTTP client for async operations
import asyncio  # Async programming utilities
from agent.audio_stream_track import CustomAudioStreamTrack  # For audio output

class OpenAIIntelligence:
    """
    This class manages real-time communication with OpenAI's API.
    It handles:
    - Audio streaming to and from OpenAI
    - Function/tool calling
    - Session management
    - Turn detection for natural conversation
    """
    def __init__(
        self, 
        loop: AbstractEventLoop,  # Event loop for async operations
        api_key,  # OpenAI API key
        model: str = "gpt-4o-realtime-preview-2024-10-01",  # Real-time model version
        instructions="""\
            Actively listen to the user's questions and provide concise, relevant responses. 
            Acknowledge the user's intent before answering. Keep responses under 2 sentences.\
        """,  # Default system prompt
        base_url: str = "api.openai.com",  # API endpoint
        voice: Voices = Voices.Alloy,  # Default voice for TTS
        temperature: float = 0.8,  # Response randomness (higher = more creative)
        tools: List[Dict[str, Union[str, any]]] = [],  # Available tools/functions
        input_audio_transcription: InputAudioTranscription = InputAudioTranscription(
            model="whisper-1"  # Model for transcribing user speech
        ),
        clear_audio_queue: Callable[[], None] = lambda: None,  # Function to clear audio when user speaks
        handle_function_call: Callable[[ResponseFunctionCallArgumentsDone], None] = lambda x: None,  # Function call handler
        modalities=["text", "audio"],  # Response types (text, audio, or both)
        max_response_output_tokens=512,  # Maximum response length
        turn_detection: ServerVADUpdateParams = ServerVADUpdateParams(
            type="server_vad",  # Server-side Voice Activity Detection
            threshold=0.5,  # VAD sensitivity
            prefix_padding_ms=300,  # Add padding before speech start
            silence_duration_ms=200,  # How long silence before turn end
        ),
        audio_track: CustomAudioStreamTrack = None,  # Audio output track
    
        ):
        # Store all constructor parameters
        self.model = model
        self.loop = loop
        self.api_key = api_key
        self.instructions = instructions
        self.base_url = base_url
        self.temperature = temperature
        self.voice = voice
        self.tools = tools
        self.modalities = modalities
        self.max_response_output_tokens = max_response_output_tokens
        self.input_audio_transcription = input_audio_transcription
        self.clear_audio_queue = clear_audio_queue
        self.handle_function_call = handle_function_call
        self.turn_detection = turn_detection
        self.ws = None  # WebSocket connection (set in connect())
        self.audio_track = audio_track
        
        # Create HTTP session for API requests
        self._http_session = aiohttp.ClientSession(loop=self.loop)
        
        # Prepare session parameters for OpenAI
        self.session_update_params = SessionUpdateParams(
            model=self.model,
            instructions=self.instructions,
            input_audio_format=AudioFormats.PCM16,  # 16-bit PCM audio input
            output_audio_format=AudioFormats.PCM16,  # 16-bit PCM audio output
            temperature=self.temperature,
            voice=self.voice,
            tool_choice="auto",  # Let OpenAI decide when to use tools
            tools=self.tools,
            turn_detection=self.turn_detection,
            modalities=self.modalities,
            max_response_output_tokens=self.max_response_output_tokens,
            input_audio_transcription=self.input_audio_transcription,
        )
        
        # Event flag to indicate when WebSocket is connected
        self.connected_event = asyncio.Event()
        
        # Store instructions if they're set before connection
        self.pending_instructions: Optional[str] = None

    async def connect(self):
        """
        Connect to OpenAI's real-time API via WebSocket.
        This initializes the communication channel for audio streaming.
        """
        # Real-time API endpoint URL
        # url = f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        url = f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        logger.info("Establishing OpenAI WS connection... ")
        
        # Connect to WebSocket with API key authentication
        self.ws = await self._http_session.ws_connect(
            url=url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",  # Required for real-time API
            },
        )
        
        # Signal that connection is ready
        self.connected_event.set()
        
        # Apply any pending instructions set before connection
        if self.pending_instructions is not None:
            await self.update_session_instructions(self.pending_instructions)

        logger.info("OpenAI WS connection established")
        
        # Start background task to handle incoming messages
        self.receive_message_task = self.loop.create_task(
            self.receive_message_handler()
        )

        print("List of tools", self.tools)

        # Initialize session with our parameters
        await self.update_session(self.session_update_params)

        # Wait for message handler to complete (runs indefinitely)
        await self.receive_message_task
        
    async def update_session_instructions(self, new_instructions: str):
        """
        Update the system instructions (prompt) during a conversation.
        
        This allows dynamically changing how the AI behaves without
        disconnecting and reconnecting.
        """
        if self.ws is None:
            # If not connected yet, store for later
            self.pending_instructions = new_instructions
            return
        
        # Update our stored parameters
        self.session_update_params.instructions = new_instructions
        # Send update to OpenAI
        await self.update_session(self.session_update_params)

    async def update_session(self, session: SessionUpdateParams):
        """
        Send updated session configuration to OpenAI.
        
        This controls things like model, temperature, available tools,
        and system instructions.
        """
        print("Updating session", session.tools)
        await self.send_request(
            SessionUpdate(
                event_id=generate_event_id(),  # Create unique ID for this event
                session=session,
            )
        )
        
    
    async def send_request(self, request: ClientToServerMessage):
        """
        Send a message to OpenAI via the WebSocket.
        
        This is the low-level method used by other methods to
        send specific types of requests.
        """
        request_json = to_json(request)  # Convert to JSON
        await self.ws.send_str(request_json)  # Send over WebSocket
        
    async def send_audio_data(self, audio_data: bytes):
        """
        Send audio data to OpenAI for transcription and processing.
        
        This is called continuously with chunks of audio data
        from the meeting participants.
        """
        # Make sure we're connected before sending
        await self.connected_event.wait()
        
        # Audio data must be base64 encoded for WebSocket
        base64_audio_data = base64.b64encode(audio_data).decode("utf-8")
        
        # Create and send the audio append message
        message = InputAudioBufferAppend(audio=base64_audio_data)
        await self.send_request(message)

    async def receive_message_handler(self):
        """
        Background task that continuously receives and processes
        messages from OpenAI.
        
        This handles all incoming events like transcription results,
        audio responses, function calls, etc.
        """
        while True:
            async for response in self.ws:
                try:
                    await asyncio.sleep(0.01)  # Small delay to prevent CPU hogging
                    
                    if response.type == aiohttp.WSMsgType.TEXT:
                        # Process text messages (JSON responses)
                        # print("Received message", response)
                        self.handle_response(response.data)
                    elif response.type == aiohttp.WSMsgType.ERROR:
                        # Log any WebSocket errors
                        logger.error("Error while receiving data from openai", response)
                except Exception as e:
                    traceback.print_exc()
                    print("Error in receiving message:", e)

    def clear_audio_queue(self):
        """
        Placeholder method to clear audio queue when user starts speaking.
        
        This would be replaced with actual implementation if needed.
        """
        pass
                
    def on_audio_response(self, audio_bytes: bytes):
        """
        Handle audio response data from OpenAI (text-to-speech).
        
        This sends the audio data to the custom audio track
        for playback to the meeting participants.
        """
        self.loop.create_task(
            self.audio_track.add_new_bytes(iter([audio_bytes]))
        )
        
    async def process_function_call(self, function_call):
        """
        Process a function/tool call from OpenAI.
        
        This is called when OpenAI decides to use one of our
        registered tools (like screen analysis).
        
        1. Executes the function
        2. Sends result back to OpenAI
        3. Creates a follow-up response
        """
        # Execute the function and get result
        result = await self.handle_function_call(function_call)
        
        print("Sending response of tool call", result)
        
        # Create message with function result
        res = ItemCreate(item=FunctionCallOutputItemParam(
            call_id=function_call.call_id,
            output=result
        ))
        
        print("tool result event id", res.event_id)
        
        # Send function result back to OpenAI
        await self.send_request(res)
        
        # Create a response to have the assistant process the result
        # This makes the AI respond to the tool's output
        response_instruction = ResponseCreate(
            response=ResponseCreateParams(
                modalities=["text", "audio"],  # Generate both text and audio
                # Tell the AI how to respond to the tool result
                instructions=f"Ask user what help is need and provide answer in 2 line based on following screen result - {result}",
                voice="alloy",
                output_audio_format="pcm16"
            )
        )
        
        # Send the instruction to OpenAI
        await self.send_request(response_instruction)
        
    def handle_response(self, message: str):
        """
        Process different types of messages from OpenAI.
        
        This handles all the different event types that can
        come from the real-time API.
        """
        # Parse JSON message
        message = json.loads(message)

        # Use pattern matching to handle different message types
        match message["type"]:
            
            case EventType.SESSION_CREATED:
                # Session was created successfully
                logger.info(f"Server Message: {message['type']}")
                # print("Session Created", message["session"])
                
            case EventType.SESSION_UPDATE:
                # Session was updated successfully
                logger.info(f"Server Message: {message['type']}")
                # print("Session Updated", message["session"])

            case EventType.RESPONSE_AUDIO_DELTA:
                # Received a chunk of audio from text-to-speech
                logger.info(f"Server Message: {message['type']}")
                self.on_audio_response(base64.b64decode(message["delta"]))

            case EventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                # Function call is ready to be processed
                logger.info(f"Server Message: {message['type']}")
                print(message)
                if not self.handle_function_call:
                    return
                    
                # Parse function call information
                function_call = ResponseFunctionCallArgumentsDone(
                    name=message["name"],
                    arguments=message.get("arguments", ""),
                    response_id=message.get("response_id",""),
                    item_id=message.get("item_id", ""),
                    output_index=message.get("output_index", ""),
                    call_id=message.get("call_id", ""),
                    event_id=message.get("event_id", "")
                )
                
                # Process function call in background
                self.loop.create_task(
                    self.process_function_call(function_call)
                )
                
            case EventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                # Transcript of the AI's spoken response
                logger.info(f"Server Message: {message['type']}")
                print(f"Response Transcription: {message['transcript']}")
            
            case EventType.ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                # Transcript of the user's speech
                logger.info(f"Server Message: {message['type']}")
                print(f"Client Transcription: {message['transcript']}")
            
            case EventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                # User has started speaking
                logger.info(f"Server Message: {message['type']}")
                logger.info("Clearing audio queue")
                self.clear_audio_queue()

        
            case EventType.ERROR:
                # Error from OpenAI
                print(message)
                logger.error(f"Server Error Message: ", message["error"])