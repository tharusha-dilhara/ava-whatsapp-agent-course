import os
import discord
from discord.ext import commands

from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ai_companion.graph import graph_builder
from ai_companion.settings import settings
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText, TextToSpeech

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

@bot.command()
async def hello(ctx):
    await ctx.send("👋 Hello! AI Companion is here.")

@bot.command(name="ask")
async def ask_ai(ctx, *, query):
    """Text-based query to AI."""
    await ctx.send("🤖 Thinking...")

    thread_id = ctx.author.id

    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)

        response_text = ""
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=query)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            if chunk[1]["langgraph_node"] == "conversation_node" and isinstance(chunk[0], AIMessageChunk):
                response_text += chunk[0].content

        await ctx.send(response_text or "🤖 No response.")

@bot.command(name="speak")
async def speak_ai(ctx):
    """Transcribe voice message and reply with AI and voice."""
    if not ctx.message.attachments:
        return await ctx.send("❌ Please upload an audio file (.mp3, .wav).")

    audio = ctx.message.attachments[0]
    audio_bytes = await audio.read()

    await ctx.send("🔊 Transcribing your voice...")

    # Transcribe
    transcription = await speech_to_text.transcribe(audio_bytes)

    # Use LangGraph for response
    thread_id = ctx.author.id
    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)
        output_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=transcription)]},
            {"configurable": {"thread_id": thread_id}},
        )

    reply = output_state["messages"][-1].content
    voice_bytes = await text_to_speech.synthesize(reply)

    await ctx.send(f"📝 Transcript: {transcription}")
    await ctx.send(f"💬 AI says: {reply}")
    await ctx.send(file=discord.File(fp=voice_bytes, filename="response.mp3"))

@bot.command(name="analyze")
async def analyze_image(ctx):
    """Analyze uploaded image using AI."""
    if not ctx.message.attachments:
        return await ctx.send("❌ Please upload an image.")

    image = ctx.message.attachments[0]
    image_bytes = await image.read()

    await ctx.send("🖼️ Analyzing image...")

    description = await image_to_text.analyze_image(
        image_bytes,
        "Please describe what you see in this image in the context of a conversation."
    )

    await ctx.send(f"📸 Image Analysis: {description}")

if __name__ == "__main__":
    if not TOKEN:
        raise ValueError("DISCORD_TOKEN not set in environment variables")
    bot.run(TOKEN)
