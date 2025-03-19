import React from "react";
import { useMeeting } from "@videosdk.live/react-sdk";
import { ParticipantView } from "./ParticipantView";
import {
  Bot,
  Mic,
  MicOff,
  MonitorUp,
  PhoneOff,
  UserPlus,
  Video,
  VideoOff,
} from "lucide-react";
import useMeetingStore from "../store/meetingStore";

interface MeetingViewProps {
  setMeetingId: (id: string | null) => void;
}

export const MeetingView: React.FC<MeetingViewProps> = ({ setMeetingId }) => {
  const {
    participants,
    localScreenShareOn,
    toggleScreenShare,
    end,
    meetingId,
    localMicOn,
    localWebcamOn,
    toggleWebcam,
    toggleMic,
  } = useMeeting();
  const { token, aiJoined, setAiJoined } = useMeetingStore();

  const inviteAI = async () => {
    try {
      const response = await fetch("http://localhost:8000/join-player", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ meeting_id: meetingId, token }),
      });

      if (!response.ok) throw new Error("Failed to invite AI");
      setAiJoined(true);
    } catch (error) {
      console.error("Error inviting AI:", error);
    }
  };

  const participantIds = Array.from(participants.keys());

  const aiParticipant = participantIds.find((id) =>
    participants.get(id)?.displayName?.toLowerCase().includes("ai")
  );

  const humanParticipant = participantIds.find(
    (id) => !participants.get(id)?.displayName?.toLowerCase().includes("ai")
  );

  const handleEndMeeting = () => {
    end();
    setMeetingId(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black p-8">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-2 gap-8">
          {humanParticipant && (
            <ParticipantView participantId={humanParticipant} />
          )}

          {aiParticipant ? (
            <ParticipantView participantId={aiParticipant} isAI />
          ) : (
            <div className="aspect-video bg-gray-800/50 rounded-xl flex items-center justify-center">
              {!aiJoined ? (
                <button
                  onClick={inviteAI}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-white"
                >
                  <UserPlus className="w-5 h-5" />
                  Invite AI Agent
                </button>
              ) : (
                <div className="flex items-center gap-3 text-gray-400">
                  <Bot className="w-8 h-8 animate-pulse" />
                  Waiting for AI to join...
                </div>
              )}
            </div>
          )}
        </div>

        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-4">
          {aiJoined && (
            <button
              onClick={() => toggleScreenShare()}
              className={`p-4 rounded-full ${
                localScreenShareOn
                  ? "bg-blue-600"
                  : "bg-gray-700 hover:bg-gray-600"
              }`}
            >
              <MonitorUp className="w-6 h-6 text-white" />
            </button>
          )}
          {/* mic */}
          <button
            onClick={() => toggleMic()}
            className={`p-4 rounded-full ${
              localMicOn ? "bg-blue-600" : "bg-gray-700 hover:bg-gray-600"
            }`}
          >
            {localMicOn ? (
              <Mic className="w-6 h-6 text-white transform rotate-225" />
            ) : (
              <MicOff className="w-6 h-6 text-white transform rotate-225" />
            )}
          </button>

          {/* webcam */}
          <button
            onClick={() => toggleWebcam()}
            className={`p-4 rounded-full ${
              localWebcamOn ? "bg-blue-600" : "bg-gray-700 hover:bg-gray-600"
            }`}
          >
            {localWebcamOn ? (
              <Video className="w-6 h-6 text-white transform rotate-225" />
            ) : (
              <VideoOff className="w-6 h-6 text-white transform rotate-225" />
            )}
          </button>

          <button
            onClick={() => handleEndMeeting()}
            className="p-4 rounded-full bg-red-500 hover:bg-red-600"
          >
            <PhoneOff className="w-6 h-6 text-white transform rotate-225" />
          </button>
        </div>
      </div>
    </div>
  );
};
