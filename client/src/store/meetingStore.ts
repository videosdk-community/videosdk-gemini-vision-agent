import { create } from 'zustand';

interface MeetingStore {
  token: string;
  aiJoined: boolean;
  setAiJoined: (joined: boolean) => void;
}

const useMeetingStore = create<MeetingStore>((set) => ({
  token: import.meta.env.VITE_VIDEOSDK_TOKEN || '',
  aiJoined: false,
  setAiJoined: (joined) => set({ aiJoined: joined }),
}));

export default useMeetingStore;