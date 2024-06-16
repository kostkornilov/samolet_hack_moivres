import json

encoded_string = "\u0441\u043a\u0438\u0434\u043a\u0430"
decoded_string = json.loads(f'"{encoded_string}"')
print(decoded_string)