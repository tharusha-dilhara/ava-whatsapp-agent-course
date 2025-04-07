import discord
from discord.ext import commands
import os

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")

@bot.command()
async def hello(ctx):
    await ctx.send("Hello! I'm alive and running in Docker 🐳")

if __name__ == "__main__":
    if not TOKEN:
        raise ValueError("Discord token not set in environment variable DISCORD_BOT_TOKEN")
    bot.run(TOKEN)
