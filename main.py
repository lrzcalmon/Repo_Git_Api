from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
#from uuid import UUID, uuid4

import ValorPossivel.ValorPossivel
from build.lib.ValorPossivel import valorpossivel


class ValorP(BaseModel):
    lista: str
    description: Optional[str] = None

app = FastAPI()
valorp = []

@app.get("/valorp/", response_model=list)
async def read_valorp(valorp):
    valorp = valorpossivel([1,2,3,4,5,6])
    return valorp

