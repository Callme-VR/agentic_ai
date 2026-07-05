# so now we are creating a graph
# and the first thing you create is states is typed dict


import os
import typing 
from typing import TypedDict

class States(TypedDict):
     # create the blue print of the states
     topic:str
     summary:str
     score:int
     
# pydantic model
# it is good in runtime and data validation and type checking

from pydantic import BaseModel , field_validator

class states(BaseModel):
     topic:str
     summary:str=""
     score:int
     
     @field_validator('score')
     def score_positive(cls,v):
          if v<0:
               raise ValueError("Score must be positive")
          return v

