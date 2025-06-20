# Audio Analysis Summarization Demo

This web application provides an AI-powered audio analysis and summarization service. It combines the Behavioral Signals API for audio analysis with ChatGPT for generating human-readable summaries.

## Features

- **Audio Upload**: Support for WAV and MP3 files
- **Audio Analysis**: Uses Behavioral Signals API to analyze:
  - Speech transcription
  - Emotion detection
  - Speaking rate analysis
  - Gender and age estimation
  - Engagement levels
  - Sentiment analysis
- **AI Summarization**: ChatGPT generates comprehensive summaries of the analysis results
- **User-friendly Interface**: Clean Streamlit web interface with real-time progress tracking

## Architecture

- **Backend**: FastAPI server that handles audio processing and API integrations
- **Frontend**: Streamlit web application for user interaction
- **APIs**: Integration with Behavioral Signals API and OpenAI ChatGPT API

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit the `config.json` file and add your OpenAI API key:

```json
{
    "openai_api_key": "your_actual_openai_api_key_here",
    "model": "gpt-3.5-turbo"
}
```

### 3. Ensure Behavioral Signals API Configuration

Make sure the main `api.config` file in the root directory is properly configured with your Behavioral Signals API credentials.

### 4. Start the Backend Server

```bash
python backend.py
```

The FastAPI backend will start on `http://localhost:8000`

### 5. Start the Frontend Application

In a new terminal window:

```bash
streamlit run frontend.py
```

The Streamlit frontend will open in your browser (usually `http://localhost:8501`)

## Usage

1. **Open the Web Interface**: Navigate to the Streamlit URL in your browser
2. **Upload Audio File**: Use the file uploader to select a WAV or MP3 file
3. **Start Analysis**: Click the "Analyze Audio" button
4. **View Results**: The application will display:
   - AI-generated summary of the audio analysis
   - Processing metrics (duration, processing time, real-time ratio)
   - Raw analysis data (expandable section)

## API Endpoints

### Backend API

- `POST /analyze-audio`: Upload and analyze audio file
- `GET /health`: Health check endpoint

## File Structure

```
other/demo_summarization/
├── backend.py          # FastAPI backend server
├── frontend.py         # Streamlit frontend application
├── config.json         # Configuration file for API keys
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

- **FastAPI**: Web framework for the backend API
- **Streamlit**: Frontend web application framework
- **OpenAI**: ChatGPT API integration
- **pydub**: Audio file format conversion
- **requests**: HTTP client for API calls

## Troubleshooting

### Backend Not Accessible
- Ensure the backend server is running (`python backend.py`)
- Check that port 8000 is not blocked by firewall
- Verify the backend URL in `frontend.py` matches your setup

### Audio Processing Errors
- Ensure your Behavioral Signals API credentials are correctly configured
- Check that the audio file format is supported (WAV or MP3)
- Verify the audio file is not corrupted

### ChatGPT API Errors
- Verify your OpenAI API key is valid and has sufficient credits
- Check the model name in `config.json` is correct
- Ensure you have access to the specified model

## Security Notes

- Keep your API keys secure and never commit them to version control
- The `config.json` file contains sensitive information and should be protected
- Consider using environment variables for production deployments

## Development

To extend or modify the application:

1. **Backend modifications**: Edit `backend.py` to add new endpoints or modify processing logic
2. **Frontend modifications**: Edit `frontend.py` to change the user interface
3. **Configuration**: Modify `config.json` to add new settings or API configurations

## License

This demo application is provided as-is for demonstration purposes.
