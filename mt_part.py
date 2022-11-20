from metaapi_cloud_sdk import MetaApi

token = ''
api = MetaApi(token=token)
account = api.metatrader_account_api.get_account_by_token()
connection = account.get_rpc_connection()
connection.connect()
connection.wait_synchronized()

symbol = 'XAUUSD'
lot = 0.01

def make_order(type):
    if type == 'Buy':
        connection.create_market_buy_order(symbol=symbol, volume=lot, take_profit=100, options={'comment': 'Py'})
        return print('Successfully BUY')
    else:
        connection.create_market_sell_order(symbol=symbol, volume=lot, take_profit=100, options={'comment': 'Py'})
        return print('Successfully SELL')