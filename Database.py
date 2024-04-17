from distutils.command.config import config
import pymongo
from pymongo import *
from datetime import datetime
import urllib.parse
import certifi

#pip install pymongo[srv]
#pip install pymongo
type_database = "SERVICE" 
if type_database == "SERVICE":
  with open("config/database_config_service.txt", "r") as file:
      config = file.read()
  list_config = config.split()
  PATH_DATABASE = list_config[0] + urllib.parse.quote(list_config[1]) + list_config[2]
else:
  with open("config/database_config.txt", "r") as file:
      PATH_DATABASE = file.read()
  
      


class Database:
  def __init__(self, PATH_DATABASE, NAME_TABLE):
    self.server = pymongo.MongoClient(PATH_DATABASE, tlsCAFile=certifi.where())
    self.db = self.server["db"]
    self.table = self.db[NAME_TABLE]
    
    ### Insert
  def insert_message(self,objection, token, message, time):
    mydict = { "objection": objection, "session_id": token, "message": message, "time": time}
    self.table.insert_one(mydict)
    
  def insert_message_user(self,session_id,current_query,current_clean_query,current_product,context_error,step_conversation,current_intent):
    now = datetime.now()
    mydict = {"session_id": session_id,"objection":"user", "query":current_query, "clean_query":current_clean_query,"product":current_product,"context_error":context_error,
              "step_conversation":step_conversation, "intent":current_intent, "time": now.strftime("%d/%m/%Y %H:%M:%S")}
    self.table.insert_one(mydict)
    
  def delete_one_message_by_token(self, token):
    myquery = { "session_id":  token}
    self.table.delete_one(myquery)
  
  ### delete
  def delete_many_message_by_token(self, token):
    myquery = { "session_id":  token}
    self.table.delete_one(myquery)

  ### find 
  def find_last_conservation_chatbot_by_token(self, token):
    mytoken = {"session_id" : token, "objection":"chatbot"}    
    return self.table.find_one(mytoken, sort=[( 'time', pymongo.DESCENDING )])
  def find_last_conservation_user_by_token(self, token):
    mytoken = {"session_id" : token, "objection":"user"}    
    return self.table.find_one(mytoken, sort=[( 'time', pymongo.DESCENDING )])
  ### update 
  
  
now = datetime.now()
# # print(now.strftime("%d/%m/%Y %H:%M:%S"))

my_database = Database(PATH_DATABASE, "database_conservation")
# my_database.insert_message("chatbot", "abcd", "chào bạn", now.strftime("%d/%m/%Y %H:%M:%S"))
# my_database.insert_message("user", "abcd", "chào bot", now.strftime("%d/%m/%Y %H:%M:%S"))
# # my_database.delete_many_message_by_token("abcd")
