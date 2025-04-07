import os
import discord
from discord.ext import commands

from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ai_companion.graph import graph_builder
from ai_companion.settings import settings

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")

@bot.command()
async def hello(ctx):
    await ctx.send("Hello! I'm alive and connected to the AI brain 🧠")

@bot.command(name="ask")
async def ask_ai(ctx, *, query):
    """Ask a question to the AI assistant."""
    await ctx.send("🤖 Thinking...")

    thread_id = ctx.author.id  # Use Discord user ID as thread

    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)

        # Stream the response in chunks
        response_text = ""
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=query)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            if chunk[1]["langgraph_node"] == "conversation_node" and isinstance(chunk[0], AIMessageChunk):
                response_text += chunk[0].content

        # Send the final message
        await ctx.send(response_text or "🤖 I had nothing to say.")

if __name__ == "__main__":
    if not TOKEN:
        raise ValueError("DISCORD_TOKEN not set in environment variables")
    bot.run(TOKEN)
