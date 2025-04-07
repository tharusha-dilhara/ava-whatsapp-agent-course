import os
import discord
from discord.ext import commands
from io import BytesIO

from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ai_companion.graph import graph_builder
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText, TextToSpeech
from ai_companion.settings import settings

# Load environment token
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# AI Modules
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")

@bot.event
async def on_message(message):
    """Handles incoming messages and status updates in Discord."""
    
    if message.author == bot.user:
        return  # Ignore the bot's own messages
    
    # Extract message content and user info
    content = message.content
    from_user = message.author.id
    session_id = str(from_user)  # Using user ID as session ID

    # Handle different types of messages
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.endswith(('.mp3', '.wav')):
                # Handle audio message
                audio_bytes = await attachment.read()
                transcription = await process_audio_message(audio_bytes)
                await handle_response(session_id, transcription, message.channel)
            elif attachment.filename.endswith(('.png', '.jpg', '.jpeg')):
                # Handle image message
                image_bytes = await attachment.read()
                description = await process_image_message(image_bytes)
                await handle_response(session_id, description, message.channel)
    else:
        # Handle text message
        await handle_response(session_id, content, message.channel)

# Processing audio message (transcription)
async def process_audio_message(audio_bytes):
    """Transcribe audio message."""
    transcription = await speech_to_text.transcribe(audio_bytes)
    return transcription

# Processing image message (image analysis)
async def process_image_message(image_bytes):
    """Analyze image message."""
    description = await image_to_text.analyze_image(
        image_bytes,
        "Please describe what you see in this image in the context of our conversation."
    )
    return description

# Handle AI response and send to Discord
async def handle_response(session_id, content, channel):
    """Process message through AI graph and send response."""
    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)

        response_text = ""
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=content)]},
            {"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            if chunk[1]["langgraph_node"] == "conversation_node" and isinstance(chunk[0], AIMessageChunk):
                response_text += chunk[0].content

        if not response_text:
            response_text = "🤖 No response."

        # Send text response
        await channel.send(response_text)

        # If the AI response is voice-based, send as audio file
        voice_bytes = await text_to_speech.synthesize(response_text)
        voice_file = BytesIO(voice_bytes)
        voice_file.seek(0)

        await channel.send(file=discord.File(fp=voice_file, filename="response.mp3"))

# Run the bot
if __name__ == "__main__":
    if not TOKEN:
        raise ValueError("DISCORD_TOKEN not set in environment variables")
    bot.run(TOKEN)
