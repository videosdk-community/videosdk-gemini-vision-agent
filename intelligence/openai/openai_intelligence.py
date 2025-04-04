import base64
import traceback
from typing import Dict, List, Union, Callable, Optional
from utils.struct.openai import (
    AudioFormats,
    ClientToServerMessage,
    EventType,
    FunctionCallOutputItemParam,
    InputAudioBufferAppend,
    InputAudioTranscription,
    ItemCreate,
    ItemParam,
    ResponseCreate,
    ResponseCreateParams,
    ResponseFunctionCallArgumentsDone,
    ServerVADUpdateParams,
    SessionUpdate,
    SessionUpdateParams,
    ToolResult,
    Voices,
    generate_event_id,
    to_json,
)
import json

from asyncio.log import logger
from asyncio import AbstractEventLoop
import aiohttp
import asyncio
from agent.audio_stream_track import CustomAudioStreamTrack

class OpenAIIntelligence:
    """
    A class to handle real-time communication with OpenAI's streaming API.
    
    This class manages a WebSocket connection to OpenAI's real-time API,
    handles sending audio data for transcription, processes responses including
    function calls, and manages text-to-speech output.
    """
    def __init__(
        self, 
        loop: AbstractEventLoop, 
        api_key,
        model: str = "gpt-4o-realtime-preview-2024-10-01",  # Default model for real-time API
        instructions="""\
            Actively listen to the user's questions and provide concise, relevant responses. 
            Acknowledge the user's intent before answering. Keep responses under 2 sentences.\
        """,  # Default system instructions
        base_url: str = "api.openai.com",
        voice: Voices = Voices.Alloy,  # Default voice for speech synthesis
        temperature: float = 0.8,  # Controls randomness in generation
        tools: List[Dict[str, Union[str, any]]] = [],  # Available function-calling tools
        input_audio_transcription: InputAudioTranscription = InputAudioTranscription(
            model="whisper-1"  # Default speech recognition model
        ),
        clear_audio_queue: Callable[[], None] = lambda: None,
        handle_function_call: Callable[[ResponseFunctionCallArgumentsDone], None] = lambda x: None,
        modalities=["text", "audio"],  # Response formats to enable
        max_response_output_tokens=512,  # Maximum response length
        turn_detection: ServerVADUpdateParams = ServerVADUpdateParams(
            type="server_vad",
            threshold=0.5,
            prefix_padding_ms=300,
            silence_duration_ms=200,
        ),  # Voice activity detection parameters
        audio_track: CustomAudioStreamTrack = None,  # For audio output
    
        ):
        """
        Initialize the OpenAI Intelligence module.
        
        Args:
            loop: AsyncIO event loop
            api_key: OpenAI API key
            model: OpenAI model to use
            instructions: System instructions for the AI
            base_url: API base URL
            voice: Voice for text-to-speech
            temperature: Generation temperature
            tools: Function-calling tools available to the model
            input_audio_transcription: Audio transcription configuration
            clear_audio_queue: Function to clear audio queue when user starts speaking
            handle_function_call: Function to handle tool calls
            modalities: Response modalities (text, audio)
            max_response_output_tokens: Maximum tokens in response
            turn_detection: VAD parameters for detecting turn-taking
            audio_track: Custom audio track for speech output
        """
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
        self.ws = None  # WebSocket connection will be set later
        self.audio_track = audio_track
        
        # Create HTTP session for API requests
        self._http_session = aiohttp.ClientSession(loop=self.loop)
        
        # Prepare session parameters
        self.session_update_params = SessionUpdateParams(
            model=self.model,
            instructions=self.instructions,
            input_audio_format=AudioFormats.PCM16,  # 16-bit PCM audio format
            output_audio_format=AudioFormats.PCM16,
            temperature=self.temperature,
            voice=self.voice,
            tool_choice="auto",  # Let the model decide when to use tools
            tools=self.tools,
            turn_detection=self.turn_detection,
            modalities=self.modalities,
            max_response_output_tokens=self.max_response_output_tokens,
            input_audio_transcription=self.input_audio_transcription,
        )
        
        # Event to indicate when WebSocket is connected and ready
        self.connected_event = asyncio.Event()
        
        # Store pending instructions if they're set before connection is established
        self.pending_instructions: Optional[str] = None

    async def connect(self):
        """
        Establish WebSocket connection to OpenAI's real-time API.
        Start message handling task and update session parameters.
        """
        # URL for real-time API connection
        # url = f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        url = f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        
        logger.info("Establishing OpenAI WS connection... ")
        
        # Connect to WebSocket with API key in headers
        self.ws = await self._http_session.ws_connect(
            url=url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",  # Required header for real-time API
            },
        )
        
        # Signal that connection is established
        self.connected_event.set()
        
        # Apply any pending instructions that were set before connection
        if self.pending_instructions is not None:
            await self.update_session_instructions(self.pending_instructions)

        logger.info("OpenAI WS connection established")
        
        # Start background task to handle incoming messages
        self.receive_message_task = self.loop.create_task(
            self.receive_message_handler()
        )

        print("List of tools", self.tools)

        # Send initial session update with configuration
        await self.update_session(self.session_update_params)

        # Wait for message handling task to complete (should run indefinitely)
        await self.receive_message_task
        
    async def update_session_instructions(self, new_instructions: str):
        """
        Dynamically update the system instructions (the system prompt) 
        for the AI assistant.
        
        Args:
            new_instructions: New system instructions to use
        """
        if self.ws is None:
            # If not connected yet, store instructions to apply after connection
            self.pending_instructions = new_instructions
            return
        
        # Update the stored session parameters
        self.session_update_params.instructions = new_instructions
        # Send the update to the API
        await self.update_session(self.session_update_params)

    async def update_session(self, session: SessionUpdateParams):
        """
        Send updated session parameters to the server.
        
        Args:
            session: Updated session parameters
        """
        print("Updating session", session.tools)
        await self.send_request(
            SessionUpdate(
                event_id=generate_event_id(),  # Generate unique ID for this event
                session=session,
            )
        )
        
    
    async def send_request(self, request: ClientToServerMessage):
        """
        Send a request to the OpenAI API over WebSocket.
        
        Args:
            request: The request to send
        """
        request_json = to_json(request)
        await self.ws.send_str(request_json)
        
    async def send_audio_data(self, audio_data: bytes):
        """
        Send audio data to OpenAI for transcription and processing.
        Waits for connection to be established before sending.
        
        Args:
            audio_data: Raw PCM audio data
        """
        await self.connected_event.wait()  # Ensure connection is established
        
        # Convert binary audio data to base64 for transmission
        base64_audio_data = base64.b64encode(audio_data).decode("utf-8")
        
        # Create and send audio buffer append message
        message = InputAudioBufferAppend(audio=base64_audio_data)
        await self.send_request(message)

    async def receive_message_handler(self):
        """
        Background task that continuously receives and processes messages
        from the OpenAI API WebSocket.
        """
        while True:
            async for response in self.ws:
                try:
                    await asyncio.sleep(0.01)  # Prevent CPU hogging
                    
                    if response.type == aiohttp.WSMsgType.TEXT:
                        # Process text messages from the server
                        # print("Received message", response)
                        self.handle_response(response.data)
                    elif response.type == aiohttp.WSMsgType.ERROR:
                        # Log errors
                        logger.error("Error while receiving data from openai", response)
                except Exception as e:
                    traceback.print_exc()
                    print("Error in receiving message:", e)

    def clear_audio_queue(self):
        """
        Empty method, meant to be overridden if needed.
        Would clear any pending audio data when user starts speaking.
        """
        pass
                
    def on_audio_response(self, audio_bytes: bytes):
        """
        Handle audio response from OpenAI (text-to-speech output).
        Sends the audio to the custom audio track for playback.
        
        Args:
            audio_bytes: Raw audio data from OpenAI TTS
        """
        self.loop.create_task(
            self.audio_track.add_new_bytes(iter([audio_bytes]))
        )
        
    async def process_function_call(self, function_call):
        """
        Process a function call from the OpenAI model.
        Executes the function and sends the result back to OpenAI.
        
        Args:
            function_call: Function call information from OpenAI
        """
        # Execute the function call and get the result
        result = await self.handle_function_call(function_call)
        
        print("Sending response of tool call", result)
        
        # Create an item with the function call output
        res = ItemCreate(item=FunctionCallOutputItemParam(
            call_id=function_call.call_id,
            output=result
        ))
        
        print("tool result event id", res.event_id)
        # Send the function result back to OpenAI
        await self.send_request(res)
        
        # Create a response to instruct the assistant to vocalize the output
        response_instruction = ResponseCreate(
            response=ResponseCreateParams(
                modalities=["text", "audio"],  # Generate both text and audio
                instructions=f"Ask user what help is need and provide answer in 2 line based on following screen result - {result}",
                voice="alloy",
                output_audio_format="pcm16"
            )
        )
        
        # Send the instruction to the assistant
        await self.send_request(response_instruction)
        
    def handle_response(self, message: str):
        """
        Process various message types from the OpenAI API.
        
        Args:
            message: JSON message string from the server
        """
        message = json.loads(message)

        # Use pattern matching to handle different message types
        match message["type"]:
            
            case EventType.SESSION_CREATED:
                # Session was successfully created
                logger.info(f"Server Message: {message['type']}")
                # print("Session Created", message["session"])
                
            case EventType.SESSION_UPDATE:
                # Session was successfully updated
                logger.info(f"Server Message: {message['type']}")
                # print("Session Updated", message["session"])

            case EventType.RESPONSE_AUDIO_DELTA:
                # Received a chunk of audio for playback
                logger.info(f"Server Message: {message['type']}")
                self.on_audio_response(base64.b64decode(message["delta"]))

            case EventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                # Function call from the model is ready to be processed
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
                
                # Process function call in the background
                self.loop.create_task(
                    self.process_function_call(function_call)
                )
                
            case EventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                # Transcript of the model's audio response
                logger.info(f"Server Message: {message['type']}")
                print(f"Response Transcription: {message['transcript']}")
            
            case EventType.ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                # Transcript of the user's audio input
                logger.info(f"Server Message: {message['type']}")
                print(f"Client Transcription: {message['transcript']}")
            
            case EventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                # User has started speaking
                logger.info(f"Server Message: {message['type']}")
                logger.info("Clearing audio queue")
                self.clear_audio_queue()

        
            case EventType.ERROR:
                # Error message from the server
                print(message)
                logger.error(f"Server Error Message: ", message["error"])