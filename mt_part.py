from metaapi_cloud_sdk import MetaApi

symbolo = 'XAUUSD'
lot = 0.01
accountID = ''

async def make_order(type):
    token = ''
    api = MetaApi(token=token)
    account = await api.metatrader_account_api.get_account(account_id = accountID)
    if account.state == 'DEPLOYING':
        print('Detected DEPLOYING state. Waiting deployed')
        await account.wait_deployed()
    elif account.state != 'DEPLOYED' and account.state != 'DEPLOYING':
        print('Deploying account')
        await account.deploy()
        print('Waiting deployed')
        await account.wait_deployed()
    else:
        print('Already deployed. Skipping')
    print('Deployed. Connecting...')
    await account.wait_connected()
    print('Account connected')
    connection = account.get_streaming_connection()
    print('Establishing connection')
    await connection.connect()
    print('Started sync')
    await connection.wait_synchronized()
    print('Subscribing to market data')
    await connection.subscribe_to_market_data(symbolo)
    print('Activating terminal state')
    terminalState = connection.terminal_state
    if type == 'Buy':
        print('Creating BUY order')
        await connection.create_market_buy_order(symbol=symbolo, volume=lot,
                                                 take_profit=terminalState.price(symbol=symbolo)['ask'] + 1.0,
                                                 options={'comment': 'Py'})
        print('Successfully BUY')
    else:
        print('Creating SELL order')
        await connection.create_market_sell_order(symbol=symbolo, volume=lot,
                                                  take_profit=terminalState.price(symbol=symbolo)['bid'] - 1.0,
                                                  options={'comment': 'Py'})
        print('Successfully SELL')
    print('Closing connection')
    await connection.close()
    print('Undeploying account')
    await account.undeploy()
    print('Done')
