from mongoDB import get_mongo_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from LLM import get_llm_response
from dateutil.parser import parse as parse_time
from pinecone import Pinecone
import os
from dotenv import load_dotenv



load_dotenv()

# Import the Pinecone library
from pinecone import Pinecone

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("PINECONE_ENVIRONMENT"))  # Use the environment variable for the index host

# Create a dense index with integrated embedding
index_name = "quickstart-py"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

dense_index = pc.Index(index_name)

import json
# DB connection
client = get_mongo_client()
db = client["expense_tracker"]
expenses_collection = db["expenses"]




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
    allow_origins=["https://ai-expense-tracker-l8reqtq8g-chaitanya-lokhandes-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#adding data to pinecone
def add_to_pinecone(data):
    try:
        # Convert ISO timestamp to readable format
        readable_time = parse_time(data["timestamp"]).strftime("%B %d, %Y at %I:%M %p")

        # Create rich text for embedding
        expense_text = (
            f" spent ₹{data['amount']} on {data['category']} "
            f"for {data['description']} using {data['paymentMethod']} on {readable_time}"
        )

        # Add to Pinecone
        
        record=[{
                "id": str(data["_id"]),
                "chunk_text": expense_text,
                "uid": data["uid"],
                "category": data["category"],
                "amount": data["amount"],
                "timestamp": data["timestamp"]
                }]
        dense_index.upsert_records("name_space",record)
        
        print("✅ Expense added to Pinecone")

    except Exception as e:
        print(f"❌ Pinecone insert failed: {str(e)}")
        # Optionally re-raise the exception if you want the API to return an error
        raise HTTPException(status_code=500, detail=f"Failed to add to Pinecone: {str(e)}")







@app.post("/add-expense")
def add_expense(expense: Expense):
    data = expense.model_dump()

    # Correct way to get current UTC time as ISO string
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    result = expenses_collection.insert_one(data)
    data["_id"] = str(result.inserted_id)

    add_to_pinecone(data)  # <-- Changed from add_to_chroma

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

        # Delete from Pinecone
        dense_index.delete(
            ids=[expense_id],
            namespace="name_space"
        ) 


        return {"message": "Expense deleted from both MongoDB and Pinecone"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting expense: {str(e)}")


# Allowed categories
ALLOWED_CATEGORIES = [
    "🍴 Food", "🥤 Drinks & Snacks", "🛺 Transport", "🚬 Addiction",
    "🧼 Groceries / Essentials", "🍕 Junk Food", "🏠 Stay / Rent",
    "🎭 Entertainment"
]

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import json

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
        - "I spent 30 on metro, uber, auto," → {{ "amount": 30, "category": "🛺 Transport", "description": "Uber ride", "paymentMethod": "UPI" }}
        - "Bought groceries for 500 using card" → {{ "amount": 500, "category": "🧼 Groceries / Essentials", "description": "Bought groceries", "paymentMethod": "Card" }}

        User message: "{query}"
        """

        extracted = get_llm_response(extract_prompt).strip()

        # If LLM says no expense found, skip to Q&A
        if extracted == "NO_EXPENSE":
            print("ℹ️ No expense extracted — switching to Q&A mode.")
        else:
            try:
                expense_data = json.loads(extracted)

                if isinstance(expense_data, dict) and "amount" in expense_data:
                    # Validate category
                    category = expense_data.get("category", "").strip()
                    if category not in ALLOWED_CATEGORIES:
                        expense_data["category"] = "Other"

                    # Fill other fields
                    expense_data["uid"] = uid
                    expense_data["timestamp"] = datetime.now(timezone.utc).isoformat()
                    expense_data["name"] = "ChatBot"

                    # Save to MongoDB
                    result = expenses_collection.insert_one(expense_data)
                    expense_data["_id"] = str(result.inserted_id)

                    # Save to Pinecone
                    add_to_pinecone(expense_data)

                    return {
                        "message": "✅ Expense added successfully!",
                        "expense": expense_data
                    }

            except Exception as e:
                print(f"⚠️ Invalid JSON from LLM: {extracted} — Error: {str(e)}")

        # Step 2: General Q&A using Pinecone search
        try:
            """ filtered_results = index.search(
                    namespace="name_space", 
                    query={
                            "inputs": {"text": query}, 
                            "top_k": 20,
                            "filter": {"uid": uid},
                        },
                    fields=["category", "chunk_text",   
                             "amount", "timestamp"],
                    ) """
            
            query_payload = {
                    "inputs": {"text": query},
                    "top_k": 10,
                    "filter": {"uid": uid}
                }
            results = index.search(
                namespace="name_space",
                query=query_payload,
                fields=["chunk_text", "category", "amount", "timestamp"]  # optional
                )
            print(results   )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ Pinecone search failed: {str(e)}")
        
        hits = results["result"]["hits"]
        if not hits:
            return {"message": "No relevant expenses found for this user."}

    # Extract chunk_text from results
        context_chunks = [
            hit["fields"].get("chunk_text", "")
            for hit in hits if "fields" in hit
        ]

        context = "\n".join(context_chunks)
        prompt = f"""
        You are a smart personal expense assistant.
        Below are some of my recent expenses:
        {context}
        Now answer this question: "{query}"
        """

        llm_response = get_llm_response(prompt)

        return {
            "results": context_chunks,
            "answer": llm_response
        }

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")



