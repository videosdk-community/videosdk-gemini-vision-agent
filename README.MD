# AI Vision Agent using VideoSDK and Google Gemini Vision API

This project integrates VideoSDK, OpenAI Realtime APIs and Gemini Vision API to analyse screenshare stream in realtime

[![Gemini AI Vision Agent](https://assets.videosdk.live/images/Gemini%20Vision%20API%20%283%29.png)](https://youtu.be/gjvJd-carXg?si=0RH8-NJ-wXpLG6hy)

### Start with the project

```sh
git clone https://github.com/videosdk-community/videosdk-gemini-vision-agent.git
```

```sh
cd videosdk-gemini-vision-agent
```

### Client Setup

1. Navigate to `client` dir:
   ```sh
   cd client
   ```
2. Make a copy of the environment configuration file:

   ```sh
   cp .env.example .env
   ```

3. Create a `.env` file in the `client` folder with:

   ```env
   VITE_VIDEOSDK_TOKEN=your_videosdk_auth_token_here
   ```

Obtain your VideoSDK Auth Token from [app.videosdk.live](https://app.videosdk.live)

### Server Setup (Python FastAPI)

Create Virtual Environment (from project root):

```sh
python -m venv .venv
```

Create a virtual environment:

Install Dependencies:

```sh
pip install -r requirements.txt
```

Create Server Environment File (in project root):

```sh
cp .env.example .env
```

Add these keys to your `.env` file:

```sh
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_api_key
```

🔑 Obtaining API Keys

- **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Gemini**: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **VideoSDK Token**: [https://app.videosdk.live](https://app.videosdk.live)

---

### ▶️ Running the Application

Start the Server (From Project Root):

```sh
uvicorn app:app
```

Start the Client (From `/client` Folder):

```sh
npm run dev
```

---

For more information, check out [docs.videosdk.live](https://docs.videosdk.live).
