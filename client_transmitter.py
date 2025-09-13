import requests
import json
import time

class Transmitter : 
    def __init__(self,server_url) : 
        self.server_url = server_url #" https://d5e33cbc658b.ngrok-free.app/is_even"
    def send_data(self):
        data = {
            'list1':[1 , 2],
            'list2': [3 , 4]
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.server_url, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        result = response.json()
        return result
