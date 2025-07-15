from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get URI from environment
uri = os.getenv("MONGODB_URI")

def get_mongo_client():
    """Returns a connected MongoClient instance for use throughout the codebase."""
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client
