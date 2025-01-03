import requests

url = 'https://eodhd.com/api/economic-events?api_token=demo&fmt=json'
data = requests.get(url).json()

print(data)