import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    print(f"initialize() failed")
    quit()

# Liste des noms courants à tester
names_to_check = ["ETHUSD", "ETHUSDm", "DXY", "USDIndex", "USDIND", "DX"]
found = []

for name in names_to_check:
    info = mt5.symbol_info(name)
    if info:
        found.append(name)

print(f"Symboles trouvés : {found}")

mt5.shutdown()
