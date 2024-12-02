import discord
from discord.ext import commands
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# Configuration
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
VECTOR_DB_FILE = 'vector_db.pkl'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Sentence Transformer model

# Initialize Sentence Transformer
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Vector DB
class VectorDatabase:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)  # 384 dimensions for MiniLM
        self.metadata = []

    def add(self, embeddings, meta):
        self.index.add(embeddings)
        self.metadata.extend(meta)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'index': self.index, 'metadata': self.metadata}, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        db = VectorDatabase()
        db.index = data['index']
        db.metadata = data['metadata']
        return db

# Bot Setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True

class IndexerBot(discord.Client):
    def __init__(self, intents, *args, **kwargs):
        super().__init__(intents=intents, *args, **kwargs)
        self.vector_db = VectorDatabase()

    async def on_ready(self):
        print(f"Logged in as {self.user}")
        await self.index_server()

    async def index_server(self):
        for guild in self.guilds:  # Iterate through all servers (guilds) the bot is in
            print(f"Indexing server: {guild.name}")
            for channel in guild.text_channels:
                print(f"Indexing channel: {channel.name}")
                try:
                    messages = [msg async for msg in channel.history(limit=None)]
                    contents = [msg.content for msg in messages if msg.content]
                    embeddings = embedder.encode(contents, convert_to_tensor=False)
                    metadata = [
                        {
                            "summary": f"{msg.content[:100]}",
                            "link": f"{msg.jump_url}",
                            "message_id": msg.id,
                            "channel_name": channel.name,
                            "channel_id": channel.id,
                            "author_name": msg.author.name,
                            "author_id": msg.author.id,
                            "timestamp": str(msg.created_at),
                            "guild_name": guild.name,
                            "guild_id": guild.id
                        }
                        for msg in messages
                    ]
                    self.vector_db.add(embeddings, metadata)
                except Exception as e:
                    print(f"Failed to index {channel.name}: {e}")
        self.vector_db.save(VECTOR_DB_FILE)
        print("Indexing completed! Shutting down bot.")
        await self.close()  # Close bot after indexing

# Start the bot
client = IndexerBot(intents=intents)
client.run(BOT_TOKEN)