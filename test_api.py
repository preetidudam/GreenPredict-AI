import urllib.request, json

BASE = "http://localhost:5000/predict"

def post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE, data=data,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

VALID = {
    "pH": 6.5, "nitrogen": 300, "phosphorus": 25, "potassium": 150,
    "organic_carbon": 0.7, "ec": 0.5, "rainfall": 850, "temperature": 28,
    "soil_type": "Loamy"
}

tests = [
    ("VALID payload",                 dict(VALID),                                           200),
    ("pH > 10  -> 400",               dict(VALID, pH=11),                                    400),
    ("Nitrogen > 700  -> 400",        dict(VALID, nitrogen=800),                             400),
    ("Phosphorus > 60  -> 400",       dict(VALID, phosphorus=70),                            400),
    ("Potassium > 400  -> 400",       dict(VALID, potassium=500),                            400),
    ("Organic Carbon > 2  -> 400",    dict(VALID, organic_carbon=3),                         400),
    ("EC > 4  -> 400",                dict(VALID, ec=5),                                     400),
    ("Non-numeric pH  -> 400",        dict(VALID, pH="abc"),                                 400),
    ("Missing nitrogen  -> 400",      {k: v for k, v in VALID.items() if k != "nitrogen"},   400),
    ("Bad soil_type  -> 400",         dict(VALID, soil_type="Clay"),                         400),
    ("pH boundary = 0  -> 200",       dict(VALID, pH=0),                                     200),
    ("pH boundary = 10  -> 200",      dict(VALID, pH=10),                                    200),
    ("EC boundary = 4  -> 200",       dict(VALID, ec=4),                                     200),
    ("Nitrogen boundary = 700 -> 200",dict(VALID, nitrogen=700),                             200),
]

all_pass = True
for name, payload, expected in tests:
    status, body = post(payload)
    passed = (status == expected)
    if not passed:
        all_pass = False
    tag = "PASS" if passed else "FAIL"
    if "error" in body:
        detail = body["error"]
    else:
        top = body.get("ranked", [{}])[0]
        detail = "Top={} ({}%)".format(top.get("plant"), top.get("probability"))
    print("  [{}] {}  -> HTTP {}  |  {}".format(tag, name, status, detail))

print()
print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED - see above")
