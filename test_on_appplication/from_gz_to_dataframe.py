import gzip
import binascii
from io import BytesIO
import json
import pandas as pd

def gz_to_df(hex_string):
    #convert from hex string to bytes
    # Vi konvertere vores hex string til bytes
    compressed_data = binascii.unhexlify(hex_string[2:])

    # vi decompresser dataen
    with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as f:
        decompressed_data = f.read()

    # konverterer det tilbage til en json string
    json_result = decompressed_data.decode('utf-8').replace("'", "\"")

    d = {key: [val] for key, val in json.loads(json_result)[0].items()}
    df = pd.DataFrame.from_dict(d)

    return df

if __name__ == '__main__':
    pass