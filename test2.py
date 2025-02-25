import requests

url = "https://flask-api-example.onrender.com/sum"
data = {"num1": 10, "num2": 15}

response = requests.post(url, json=data)
print(response.json())  # Output: {'sum': 25}
