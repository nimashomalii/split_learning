import requests
import json
import time

class Transmitter : 
    def __init__(self,server_url) : 
        self.server_url = server_url #" https://d5e33cbc658b.ngrok-free.app/is_even"
    def send_data(self , x , label) :  #this method will send the output of the first neural network to the server 
        x_list = x.detach().cpu().tolist()
        label_list = label.detach().cpu().tolist()
        data  = {'value' : x_list ,  
                 'label' : label_list }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.server_url, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        result = response.json()
        return result
