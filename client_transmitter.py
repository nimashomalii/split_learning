import requests
import json
import time
import torch 

class Transmitter : 
    def __init__(self,server_url) : 
        self.server_url = server_url #" https://d5e33cbc658b.ngrok-free.app/is_even"

    def send_data(self , x , label ,device ,  status  ): #status if is train or test 
        x = x.detach().cpu().tolist()
        label = label.detach().cpu().tolist()
        if status == 'train' : 
            data = {
                'prediction_iput':x,
                'label': label ,
                'status' : status
            }
        elif status == 'test' : 
            data = {
                'prediction_iput':x,
                'status' : status
            }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.server_url, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        result = response.json()
        if status == 'train' : # result = {'grad' :  }
            grad = result['grad']
            return torch.tensor(grad).to(device)
        elif status == 'test' :
            prediction = torch.tensor(result['prediction']).to(device)
            return  prediction

