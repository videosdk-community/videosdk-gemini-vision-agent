import React from "react";
import { MeetingProvider } from "@videosdk.live/react-sdk";
import { Bot } from "lucide-react";
import { MeetingView } from "./components/MeetingView";
import useMeetingStore from "./store/meetingStore";

function App() {
  const [meetingId, setMeetingId] = React.useState<string | null>(null);
  const [userName, setUserName] = React.useState("");
  const [showNameDialog, setShowNameDialog] = React.useState(false);
  const { token } = useMeetingStore();

  const createMeeting = async () => {
    try {
      const response = await fetch("https://api.videosdk.live/v2/rooms", {
        method: "POST",
        headers: {
          Authorization: token,
          "Content-Type": "application/json",
        },
      });

      const { roomId } = await response.json();
      setShowNameDialog(true);
      setMeetingId(roomId);
    } catch (error) {
      console.error("Error creating meeting:", error);
    }
  };

  const startMeeting = () => {
    if (!userName.trim()) return;
    setShowNameDialog(false);
  };

  if (showNameDialog) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-4">
        <div className="bg-gray-800/50 backdrop-blur-sm p-8 rounded-2xl shadow-xl max-w-md w-full">
          <h2 className="text-2xl font-bold text-white mb-6">
            Enter Your Name
          </h2>
          <input
            type="text"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            placeholder="Your name"
            className="w-full px-4 py-3 bg-gray-700 rounded-lg text-white mb-4"
          />
          <button
            onClick={startMeeting}
            className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-white"
          >
            Join Meeting
          </button>
        </div>
      </div>
    );
  }

  if (!meetingId) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-4">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-8 flex items-center justify-center gap-3">
            <Bot className="w-12 h-12 text-blue-400" />
            Explore Vision AI with VideoSDK
          </h1>
          <button
            onClick={createMeeting}
            className="px-8 py-4 bg-blue-600 hover:bg-blue-700 rounded-xl text-white text-lg font-medium transition-all hover:scale-105"
          >
            Start Meeting
          </button>
        </div>
      </div>
    );
  }

  return (
    <MeetingProvider
      config={{
        meetingId,
        micEnabled: true,
        webcamEnabled: true,
        name: userName,
        debugMode: true,
      }}
      token={token}
      joinWithoutUserInteraction
    >
      <MeetingView setMeetingId={setMeetingId} />
    </MeetingProvider>
  );
}

export default App;
