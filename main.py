from mongoDB import get_mongo_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from bson import ObjectId
import chromadb
from sentence_transformers import SentenceTransformer
from LLM import get_llm_response
from dateutil.parser import parse as parse_time
# DB connection
client = get_mongo_client()
db = client["expense_tracker"]
expenses_collection = db["expenses"]

# ChromaDB client
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.get_or_create_collection(name="expenses")



model = SentenceTransformer("paraphrase-MiniLM-L3-v2")



# FastAPI app
app = FastAPI()

# Model
class Expense(BaseModel):
    amount: float
    category: str
    description: str
    paymentMethod: str
    uid: str
    name: Optional[str] = None  # <-- Add this line
    timestamp: Optional[str] = None

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-expense-tracker-nm0txqk8e-chaitanya-lokhandes-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to add to ChromaDB
def add_to_chroma(data):
    try:
        # Convert ISO timestamp to readable format
        readable_time = parse_time(data["timestamp"]).strftime("%B %d, %Y at %I:%M %p")

        # Create rich text for embedding
        expense_text = (
            f"{data['name']} spent ₹{data['amount']} on {data['category']} "
            f"for {data['description']} using {data['paymentMethod']} on {readable_time}"
        )

        # Generate embedding
        embedding = model.encode([expense_text])[0]

        # Add to ChromaDB
        collection.add(
            documents=[expense_text],
            embeddings=[embedding.tolist()],
            ids=[data["_id"]],
            metadatas=[
                {
                    "uid": data["uid"],
                    "category": data["category"],
                    "amount": data["amount"],
                    "timestamp": data["timestamp"],
                }
            ],
        )

        print("✅ Expense added to ChromaDB")

    except Exception as e:
        print(f"❌ ChromaDB insert failed: {e}")


# Add expense endpoint
@app.post("/add-expense")
def add_expense(expense: Expense):
    data = expense.model_dump()

    if not data.get("timestamp"):
        data["timestamp"] = datetime.utcnow().isoformat()

    result = expenses_collection.insert_one(data)
    data["_id"] = str(result.inserted_id)

    add_to_chroma(data)

    return {"message": "Expense saved and embedded", "expense": data}

# Example FastAPI endpoint
@app.get("/expenses")
def get_expenses(uid: str):
    expenses = list(expenses_collection.find({"uid": uid}))
    # Convert ObjectId to str for each expense
    for exp in expenses:
        exp["_id"] = str(exp["_id"])
    return {"expenses": expenses}

@app.delete("/expense_del/{expense_id}")
def delete_expense(expense_id: str):
    try:
        # Delete from MongoDB
        result = expenses_collection.delete_one({"_id": ObjectId(expense_id)})
        if result.deleted_count != 1:
            raise HTTPException(status_code=404, detail="Expense not found in MongoDB")

        # Delete from ChromaDB
        collection.delete(ids=[expense_id])

        return {"message": "Expense deleted from both MongoDB and ChromaDB"}
    
    except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error deleting expense: {str(e)}")


@app.post("/chat")
async def chat_with_model(query: str):
    try:
        query_embedding = model.encode([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=50,
            include=["documents", "metadatas"]
        )

        if not results["documents"]:
            return {"message": "No relevant expenses found."}

        context = "\n".join(results["documents"][0])

        prompt = f"""
                    You are a smart personal expense assistant.
                    Below are some of my recent expenses:
                    {context}
                    Now answer this question: "{query}"
                 """

        llm_response = get_llm_response(prompt)

        return {
            "results": results["documents"][0],
            "answer": llm_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
