import pandas as pd
from pyzipcode import ZipCodeDatabase

info = pd.read_csv('u.user', names=['id', 'age', 'gender', 'job', 'zip'], sep='|')

zcdb = ZipCodeDatabase()

zipcodes = set()

for zipcode in info['zip'].unique():
    try:
        res = zcdb[zipcode].timezone
        zipcodes.add(res)
    except:
        print(zipcode)
    continue

print(zipcodes)
print("done")