import urllib.request

try:
    data = urllib.request.urlopen('http://127.0.0.1:5000/api/summary', timeout=20).read().decode()
    print(data)
except Exception as e:
    print('ERR', repr(e))
