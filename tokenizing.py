import re
text = "Hello, world. This, is a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#print(result)
result = [item for item in result if item.strip()]
print(result)


