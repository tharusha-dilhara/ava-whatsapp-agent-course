import os
import io
import logging
import httpx
import discord
from discord.ext import commands

# Import your application modules and settings
from langchain_core.messages import HumanMessage
from ai_companion.graph import graph_builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ai_companion.modules.speech import SpeechToText, TextToSpeech
from ai_companion.modules.image import ImageToText
from ai_companion.settings import settings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize multimodal modules (they can be reused across messages)
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()

# Set up Discord intents and bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Helper: Download file from a URL
async def download_file(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

# Process a Discord message and its attachments
async def process_discord_message(message: discord.Message):
    # Avoid processing messages from bots
    if message.author.bot:
        return

    combined_text = message.content.strip() if message.content else ""

    # Process attachments: Check file extension to decide which module to use
    for attachment in message.attachments:
        file_name = attachment.filename.lower()
        try:
            file_bytes = await download_file(attachment.url)
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {e}")
            continue

        # If the attachment is audio, transcribe it
        if any(file_name.endswith(ext) for ext in [".mp3", ".wav", ".ogg"]):
            try:
                logger.info(f"Transcribing audio file: {file_name}")
                transcription = await speech_to_text.transcribe(file_bytes)
                combined_text += f"\n[Audio Transcription: {transcription}]"
            except Exception as e:
                logger.error(f"Audio transcription failed for {file_name}: {e}")

        # If the attachment is an image, analyze it
        elif any(file_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif"]):
            try:
                logger.info(f"Analyzing image file: {file_name}")
                description = await image_to_text.analyze_image(
                    file_bytes,
                    "Please describe what you see in this image in the context of our conversation.",
                )
                combined_text += f"\n[Image Analysis: {description}]"
            except Exception as e:
                logger.error(f"Image analysis failed for {file_name}: {e}")

        # For other file types, you might add additional processing if needed.

    # If there is no meaningful input after processing, ignore the message.
    if not combined_text:
        return

    # Use the author's Discord ID as a session identifier
    session_id = str(message.author.id)

    try:
        # Process the combined message through the graph agent
        async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
            graph = graph_builder.compile(checkpointer=short_term_memory)
            await graph.ainvoke(
                {"messages": [HumanMessage(content=combined_text)]},
                {"configurable": {"thread_id": session_id}},
            )
            output_state = await graph.aget_state(config={"configurable": {"thread_id": session_id}})
    except Exception as e:
        logger.error(f"Graph agent processing failed: {e}", exc_info=True)
        await message.channel.send("An error occurred while processing your request.")
        return

    # Retrieve the workflow type and response from the graph agent's state
    workflow = output_state.values.get("workflow", "conversation")
    response_message = output_state.values["messages"][-1].content

    # Depending on the workflow, send the appropriate type of response
    if workflow == "audio":
        try:
            # Convert text response to audio using text_to_speech
            audio_buffer = output_state.values.get("audio_buffer")
            if not audio_buffer:
                # If the graph did not generate an audio buffer, synthesize it now
                audio_bytes = await text_to_speech.synthesize(response_message)
            else:
                audio_bytes = audio_buffer
            # Send the audio as a Discord file
            audio_file = discord.File(fp=io.BytesIO(audio_bytes), filename="response.mp3")
            await message.channel.send(content="Here is your audio response:", file=audio_file)
        except Exception as e:
            logger.error(f"Failed to send audio response: {e}", exc_info=True)
            await message.channel.send("An error occurred while sending the audio response.")

    elif workflow == "image":
        try:
            # For an image workflow, load the image file generated by the agent
            image_path = output_state.values.get("image_path")
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                image_file = discord.File(fp=io.BytesIO(image_bytes), filename="response.png")
                await message.channel.send(content=response_message, file=image_file)
            else:
                await message.channel.send("No image was generated.")
        except Exception as e:
            logger.error(f"Failed to send image response: {e}", exc_info=True)
            await message.channel.send("An error occurred while sending the image response.")
    else:
        # For text responses, just send the message
        await message.channel.send(response_message)

@bot.event
async def on_ready():
    logger.info(f"Discord bot logged in as {bot.user}")

# Listen to all messages (or you can restrict to commands if desired)
@bot.event
async def on_message(message: discord.Message):
    # Process incoming messages (ignore messages from the bot itself)
    await process_discord_message(message)
    # Allow commands (if any) to be processed as well
    await bot.process_commands(message)

if __name__ == "__main__":
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN environment variable is not set!")
    else:
        bot.run(DISCORD_TOKEN)
