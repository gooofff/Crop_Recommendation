import requests

url = "https://crop-recommendation-4sjz.onrender.com/recommendation"
data = {"N": 90, "P": 42,"K": 43,"temperature": 20.87974371,"humidity": 82.00274423,"ph": 6.502985292000001,"rainfall": 202.9355362 }

response = requests.post(url, json=data)
print(response.json())  # Output: {'sum': 25}
