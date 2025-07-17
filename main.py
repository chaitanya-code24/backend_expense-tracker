from mongoDB import get_mongo_client
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from fastapi.encoders import jsonable_encoder
from bson import ObjectId
import chromadb
from sentence_transformers import SentenceTransformer
from LLM import get_llm_response
from dateutil.parser import parse as parse_time

import json
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
    allow_origins=["https://ai-expense-tracker-l66devfl0-chaitanya-lokhandes-projects.vercel.app"],
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






@app.post("/add-expense")
def add_expense(expense: Expense):
    data = expense.model_dump()

    # Correct way to get current UTC time as ISO string
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

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


# Allowed categories
ALLOWED_CATEGORIES = [
    "🍴 Food", "🥤 Drinks & Snacks", "🛺 Transport", "🚬 Addiction",
    "🧼 Groceries / Essentials", "🍕 Junk Food", "🏠 Stay / Rent",
    "🎭 Entertainment"
]

@app.post("/chat")
async def chat_with_model(query: str, uid: str):
    try:
        # Step 1: Use LLM to extract expense if present
        extract_prompt = f"""
        Extract expense data from the user message. If no expense is found, respond only with 'NO_EXPENSE'.

        Allowed categories: "🍴 Food", "🥤 Drinks & Snacks", "🛺 Transport", "🚬 Addiction", "🧼 Groceries / Essentials", "🍕 Junk Food", "🏠 Stay / Rent", "🎭 Entertainment"

        Response JSON format:
        {{
            "amount": float,
            "category": string,
            "description": string,
            "paymentMethod": string
        }}

        Make sure the category exactly matches one from the list above.

        Examples:
        - "I spent 30 on Uber" → {{ "amount": 30, "category": "🛺 Transport", "description": "Uber ride", "paymentMethod": "UPI" }}
        - "Bought groceries for 500 using card" → {{ "amount": 500, "category": "🧼 Groceries / Essentials", "description": "Bought groceries", "paymentMethod": "Card" }}

        User message: "{query}"
        """

        extracted = get_llm_response(extract_prompt)

        try:
            expense_data = json.loads(extracted)

            if isinstance(expense_data, dict) and "amount" in expense_data:
                # Validate category
                category = expense_data.get("category", "").strip()
                if category not in ALLOWED_CATEGORIES:
                    expense_data["category"] = "other"

                # Fill additional required fields
                expense_data["uid"] = uid
                expense_data["timestamp"] = datetime.now(timezone.utc).isoformat()
                expense_data["name"] = "ChatBot"

                # Save to MongoDB
                result = expenses_collection.insert_one(expense_data)
                expense_data["_id"] = str(result.inserted_id)

                # Save to ChromaDB
                add_to_chroma(expense_data)

                return {
                    "message": "✅ Expense added successfully!",
                    "expense": expense_data
                }
        except Exception:
            pass  # Extraction failed or not JSON, continue to Q&A

        # Step 2: Regular QnA if no expense detected
        query_embedding = model.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=50,
            include=["documents", "metadatas"]
        )

        filtered_docs = [
            doc for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            if meta.get("uid") == uid
        ]

        if not filtered_docs:
            return {"message": "No relevant expenses found for this user."}

        context = "\n".join(filtered_docs)
        prompt = f"""
        You are a smart personal expense assistant.
        Below are some of my recent expenses:
        {context}
        Now answer this question: "{query}"
        """

        llm_response = get_llm_response(prompt)

        return {
            "results": filtered_docs,
            "answer": llm_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
