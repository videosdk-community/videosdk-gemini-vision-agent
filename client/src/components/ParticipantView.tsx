import React from 'react';
import { useParticipant } from '@videosdk.live/react-sdk';
import { Bot, Mic, MicOff, User } from 'lucide-react';

interface ParticipantViewProps {
  participantId: string;
  isAI?: boolean;
}

export const ParticipantView: React.FC<ParticipantViewProps> = ({ participantId, isAI }) => {
  const {
    displayName,
    isLocal,
    webcamStream,
    micStream,
    webcamOn,
    micOn,
    isActiveSpeaker,
    screenShareStream,
    screenShareOn
  } = useParticipant(participantId);

  const videoRef = React.useRef<HTMLVideoElement>(null);
  const audioRef = React.useRef<HTMLAudioElement>(null);

  React.useEffect(() => {
    if (videoRef.current && webcamStream && webcamOn) {
      const mediaStream = new MediaStream();
      mediaStream.addTrack(webcamStream.track);
      videoRef.current.srcObject = mediaStream;
      videoRef.current.play().catch(console.error);
    }
  }, [webcamStream, webcamOn]);

  React.useEffect(() => {
    if (audioRef.current && micStream && micOn) {
      const mediaStream = new MediaStream();
      mediaStream.addTrack(micStream.track);
      audioRef.current.srcObject = mediaStream;
      audioRef.current.play().catch(console.error);
    }
  }, [micStream, micOn]);

  return (
    <div className={`relative rounded-xl overflow-hidden bg-gray-800/50 backdrop-blur-sm ${isActiveSpeaker ? 'ring-2 ring-green-500' : ''}`}>
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-800/90">
        {micOn ? (
          <Mic className="w-4 h-4 text-green-400" />
        ) : (
          <MicOff className="w-4 h-4 text-gray-400" />
        )}
      </div>

      <div className="aspect-video">
        {webcamOn && webcamStream ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted={isLocal}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
            {isAI ? (
              <Bot className="w-16 h-16 text-blue-400" />
            ) : (
              <User className="w-16 h-16 text-gray-400" />
            )}
          </div>
        )}
      </div>

      <audio ref={audioRef} autoPlay playsInline muted={isLocal} />

      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
        <div className="flex items-center space-x-3">
          {isAI ? (
            <Bot className="w-5 h-5 text-blue-400" />
          ) : (
            <User className="w-5 h-5 text-gray-400" />
          )}
          <span className="text-white font-medium">
            {displayName} {isLocal && "(You)"}
          </span>
        </div>
      </div>
    </div>
  );
};