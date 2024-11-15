from fastapi import FastAPI, HTTPException
from flask import Flask
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
import uvicorn
 
app = FastAPI()
flaskapp = Flask(__name__)

class Item(BaseModel):
    text:str = None
    is_done: bool = False


items = []
 
@app.get("/app")
def main_app():
    return "Main app called!"
 
@flaskapp.route("/app2")
def flask_app():
    return "Flask app called!"
 
app.mount("/flask", WSGIMiddleware(flaskapp))

#curl -X POST -H "Content-Type: application/json" 'http://127.0.0.1:8000/items?item=apple'

@app.post("/items")
def create_iteam(item: str):
    items.append(item)
    return item

# curl -X GET -H "Content-Type: application/json" 'http://127.0.0.1:8000/items?limit=2'
@app.get("/items")
def list_iteams(limit: int = 10):
    return items[0:limit]

# curl -X GET -H "Content-Type: application/json" 'http://127.0.0.1:8000/items/1'
@app.get("/items/{item_id}")
def get_iteam(item_id: int) -> str:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"Item {item_id} Not Found")

# curl -X POST -H "Content-Type: application/json" -d '{"text": "apple"}' 'http://127.0.0.1:8000/pydantic/items'
@app.post("/pydantic/items")
def create_iteam(item: Item):
    items.append(item)
    return item

@app.get("/pydantic/items", response_model=list[Item])
def list_iteams(limit: int = 10):
    return items[0:limit]

# curl -X GET -H "Content-Type: application/json" 'http://127.0.0.1:8000/pydantic/items/0'
@app.get("/pydantic/items/{item_id}", response_model=Item)
def get_iteam(item_id: int) -> Item:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"Item {item_id} Not Found")
    


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

# To run this api on cmd prompt type: uvicorn main:app