import urllib.request
import json

data =  {
  "Inputs": {
    "data": [
      {
        "host_response_rate": 100,
        "host_identity_verified": 0,
        "host_total_listings_count": 1,
        "is_location_exact": 1,
        "property_type": 0,
        "accommodates": 2,
        "price": 5000,
        "minimum_nights": 15,
        "number_of_reviews": 0,
        "review_scores_rating": 100.0,
        "instant_bookable": 1,
        "cancellation_policy": 2,
        "reviews_per_month": 1.0
      }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}
body = str.encode(json.dumps(data))

url = "http://cefae4be-6f92-4f49-b006-17c1aef41e0e.westeurope.azurecontainer.io/score"

headers = {'Content-Type':'application/json'}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))