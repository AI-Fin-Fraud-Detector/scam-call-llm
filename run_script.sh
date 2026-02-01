#for test normal, output should be False
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi, what time is our meeting tomorrow? Reply: 2pm."}'

#for test scam, output should be True
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a prize! Please provide your bank account and password, we will transfer the money to you."}'
