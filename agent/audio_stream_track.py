import asyncio
from fractions import Fraction  # For representing time base as a fraction
import threading
from time import time
import traceback
from typing import Iterator, Optional
from av import AudioFrame  # PyAV library for handling audio/video frames
import numpy as np
from videosdk import CustomAudioTrack  # Base class from the video SDK


AUDIO_PTIME = 0.02  # Packet time in seconds (20ms chunks - standard for most audio processing)


class MediaStreamError(Exception):
    """Custom exception for media stream errors"""
    pass


class CustomAudioStreamTrack(CustomAudioTrack):
    """
    A custom implementation of an audio track for streaming audio in a video meeting.
    This class provides functionality to buffer audio data, process it in chunks,
    and deliver it in a continuous stream for real-time communication.
    """
    def __init__(
        self, loop, handle_interruption: Optional[bool] = True
    ):
        """
        Initialize the custom audio track.
        
        Args:
            loop: The asyncio event loop to use for async operations
            handle_interruption: Whether to handle interruptions to the audio stream
        """
        super().__init__()
        self.loop = loop  # Store the event loop
        self._start = None  # Track when the stream starts
        self._timestamp = 0  # Current timestamp in samples
        self.frame_buffer = []  # Buffer to hold audio frames ready to be sent
        self.audio_data_buffer = bytearray()  # Raw audio data buffer
        self.frame_time = 0  # Current frame time (used for pts calculation)
        self.sample_rate = 24000  # Audio sample rate in Hz
        self.channels = 1  # Mono audio
        self.sample_width = 2  # 16-bit audio (2 bytes per sample)
        self.time_base_fraction = Fraction(1, self.sample_rate)  # Time base for frame timing
        self.samples = int(AUDIO_PTIME * self.sample_rate)  # Number of samples per frame
        self.chunk_size = int(self.samples * self.channels * self.sample_width)  # Size of each audio chunk in bytes
        self._process_audio_task_queue = asyncio.Queue()  # Queue for audio processing tasks
        
        # Start a separate thread for processing audio data
        self._process_audio_thread = threading.Thread(target=self.run_process_audio)
        self._process_audio_thread.daemon = True  # Thread will exit when main program exits
        self._process_audio_thread.start()
        
        self.skip_next_chunk = False  # Flag to skip processing the next chunk (used during interruptions)

    def interrupt(self):
        """
        Interrupt the current audio playback.
        Clears buffers and skips any pending chunks to stop current audio immediately.
        """
        length = len(self.frame_buffer)
        self.frame_buffer.clear()  # Clear all pending audio frames
        
        # Empty the task queue to stop processing more audio
        while not self._process_audio_task_queue.empty():
            self.skip_next_chunk = True
            self._process_audio_task_queue.get_nowait()
            self._process_audio_task_queue.task_done()

        if length > 0:
            self.skip_next_chunk = True

    async def add_new_bytes(self, audio_data_stream: Iterator[bytes]):
        """
        Add new audio data to be processed and played.
        
        Args:
            audio_data_stream: Iterator yielding chunks of audio data as bytes
        """
        # self.interrupt()  # Commented out, but would interrupt current playback
        await self._process_audio_task_queue.put(audio_data_stream)

    def run_process_audio(self):
        """
        Entry point for the audio processing thread.
        Runs the _process_audio coroutine in a separate thread.
        """
        asyncio.run(self._process_audio())

    async def _process_audio(self):
        """
        Main audio processing loop.
        Continuously processes audio data from the queue and creates audio frames.
        """
        while True:
            try:
                # This commented block would update character state when queue is empty
                # if (self._process_audio_task_queue.empty()) and (
                #     self.update_character_state is not None
                # ):
                    while True:
                        if len(self.frame_buffer) > 0:
                            await asyncio.sleep(0.1)
                            continue
                        # asyncio.run_coroutine_threadsafe(
                        #     self.update_character_state(
                        #         CharacterState.CHARACTER_LISTENING
                        #     ),
                        #     self.loop,
                        # )
                        break
            except Exception as e:
                print("Error while updating chracter state", e)

            try:
                # Get the next audio data stream from the queue
                audio_data_stream = asyncio.run_coroutine_threadsafe(
                    self._process_audio_task_queue.get(), self.loop
                ).result()
                
                # Process each chunk of audio data in the stream
                for audio_data in audio_data_stream:
                    try:
                        # Skip chunk handling (commented out)
                        # if self.skip_next_chunk:
                        #     print("Skipping Next Chunk")
                        #     self.frame_buffer.clear()
                        #     self.skip_next_chunk = False
                        #     break
                        
                        # Add the new audio data to the buffer
                        self.audio_data_buffer += audio_data
                        
                        # Process complete chunks from the buffer
                        while len(self.audio_data_buffer) >= self.chunk_size:
                            # Extract a complete chunk
                            chunk = self.audio_data_buffer[: self.chunk_size]
                            # Keep the remaining data in the buffer
                            self.audio_data_buffer = self.audio_data_buffer[
                                self.chunk_size :
                            ]
                            # Convert chunk to audio frame and add to frame buffer
                            audio_frame = self.buildAudioFrames(chunk)
                            self.frame_buffer.append(audio_frame)

                        # Update character state (commented out)
                        # if self.update_character_state is not None:
                            # await self.update_character_state(
                            #     CharacterState.CHARACTER_SPEAKING
                            # )
                    except Exception as e:
                        print("Error while putting audio data stream", e)
            except Exception as e:
                traceback.print_exc()
                print("Error while process audio", e)

    def buildAudioFrames(self, chunk: bytes) -> AudioFrame:
        """
        Convert raw audio bytes to an AudioFrame object.
        
        Args:
            chunk: Raw audio data as bytes
            
        Returns:
            AudioFrame: A PyAV AudioFrame object ready for streaming
        """
        # Convert bytes to numpy array of 16-bit integers
        data = np.frombuffer(chunk, dtype=np.int16)
        data = data.reshape(-1, 1)  # Reshape to (samples, channels)
        
        # Create an AudioFrame from the numpy array
        audio_frame = AudioFrame.from_ndarray(data.T, format="s16", layout="mono")
        return audio_frame

    def next_timestamp(self):
        """
        Calculate the next presentation timestamp (pts) for an audio frame.
        
        Returns:
            tuple: (pts, time_base) for the next frame
        """
        pts = int(self.frame_time)
        time_base = self.time_base_fraction
        # Increment frame time by number of samples
        self.frame_time += self.samples
        return pts, time_base

    async def recv(self) -> AudioFrame:
        """
        Receive the next audio frame for streaming.
        This method is called by the video SDK to get frames to stream.
        
        Returns:
            AudioFrame: The next audio frame to be streamed
            
        Raises:
            MediaStreamError: If the stream is not live
        """
        try:
            # Check if the stream is active
            if self.readyState != "live":
                raise MediaStreamError

            # Initialize start time on first call
            if self._start is None:
                self._start = time()
                self._timestamp = 0
            else:
                # Increment timestamp by number of samples
                self._timestamp += self.samples

            # Calculate how long to wait to maintain correct playback timing
            wait = self._start + (self._timestamp / self.sample_rate) - time()

            # Wait if needed to maintain timing
            if wait > 0:
                await asyncio.sleep(wait)

            # Get the next timestamp
            pts, time_base = self.next_timestamp()

            # Get frame from buffer or create silent frame if buffer is empty
            if len(self.frame_buffer) > 0:
                frame = self.frame_buffer.pop(0)
            else:
                # Create a silent frame (filled with zeros)
                frame = AudioFrame(format="s16", layout="mono", samples=self.samples)
                for p in frame.planes:
                    p.update(bytes(p.buffer_size))

            # Set frame timing properties
            frame.pts = pts
            frame.time_base = time_base
            frame.sample_rate = self.sample_rate
            return frame
        except Exception as e:
            traceback.print_exc()
            print("error while creating tts->rtc frame", e)