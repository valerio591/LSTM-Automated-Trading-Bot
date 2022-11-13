
# Import libraries for the query
import time
import hmac
from requests import Request, Session, Response
import requests
import urllib


class FtxClient:
    ENDPOINT = 'https://ftx.com/api/'
    
    # Initialize the class with auth info
    def __init__(self, api_key: str, api_secret: str, subaccount_name = None):
        self.session = Session()
        self.api_key = api_key
        self.api_secret = api_secret
        self.subaccount_name = subaccount_name
        
    #----------------------------------------------------------------------------------------------------------#
    #------------------------------------------- BASIC METHODS ------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#    
    
    # Define get request method
    def get(self, path: str, params: dict = None):
        return self.request('GET', path, params=params)
    
    # Define post request method
    def post(self, path: str, params: dict = None):
        return self.request('POST', path, json=params)
    
    def delete(self, path: str, params: dict = None):
        return self.request('DELETE', path, json=params)

    def request(self, method: str, path: str, **kwargs):
        request = Request(method, self.ENDPOINT + path, **kwargs)
        print(self.ENDPOINT + path)
        self.sign_request(request)
        response = self.session.send(request.prepare())
        return self.process_response(response)

    def sign_request(self, request: Request) -> None:
        ts = int(time.time() * 1000)
        prepared = request.prepare()
        signature_payload = f'{ts}{prepared.method}{prepared.path_url}'.encode()
        if prepared.body:
            signature_payload += prepared.body
        signature = hmac.new(self.api_secret.encode(), signature_payload, 'sha256').hexdigest()
        request.headers['FTX-KEY'] = self.api_key
        request.headers['FTX-SIGN'] = signature
        request.headers['FTX-TS'] = str(ts)
        if self.subaccount_name:
            request.headers['FTX-SUBACCOUNT'] = urllib.parse.quote(self.subaccount_name)

    def process_response(self, response: Response):
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        else:
            if not data['success']:
                raise Exception(data['error'])
            return data['result']
    
    #----------------------------------------------------------------------------------------------------------#
    #------------------------------------------ ORDERS METHODS ------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    
    def place_order(self, market: str, side: str, price: float, size: float, client_id: str,
                    type: str = 'limit', reduce_only: bool = False, ioc: bool = False, post_only: bool = False,
                    ) -> dict:
        return self.post('orders', {'market': market,
                                     'side': side,
                                     'price': price,
                                     'size': size,
                                     'type': type,
                                     'reduceOnly': reduce_only,
                                     'ioc': ioc,
                                     'postOnly': post_only,
                                     'clientId': client_id,
                                     })
    
    def get_orders_placed(self, market: str = 'BTC-PERP'):
        return self.get('orders?market=' + market)
    
    def get_order_status(self, order_id: int):
        return self.get('orders/' + str(order_id))    
    
    def modify_order(self, price: float, size: float, client_id: str, order_id: int) -> dict:
        return self.post('orders/' + str(order_id) + '/modify', {'price': price,
                                                            'size': size,
                                                            'clientId': client_id})
    
    def fills(self, market: str, start_time: int, end_time: int, order_id: int, order: str = None) -> dict:
        '''The order parameter can be set to "asc" to get orders filled through time in ascending order'''
        return self.get('fills', {'market': market,
                                   'start_time': start_time,
                                   'end_time': end_time,
                                   'order': order,
                                   'orderId': order_id})            
    
    def get_open_orders(self, order_id: int, market: str = None):
        return self.get(f'orders', {'market': market, 'order_id':order_id})
    
    def cancel_order(self, order_id: int):
        return self.delete('orders/' + str(order_id))
    
    def get_positions(self, showAvgPrice: bool = False, ticker: str = 'BTC-PERP'):
        futures_list = self.get('positions', {'showAvgPrice': showAvgPrice})
        positions_for_ticker = [i for i in futures_list if i['future'] == ticker]
        return positions_for_ticker[0]
    
    #----------------------------------------------------------------------------------------------------------#
    #------------------------------------------ POSITIONS METHODS ---------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#

    def get_balances(self):
        return self.get('wallet/balances')
    
    #----------------------------------------------------------------------------------------------------------#
    #------------------------------------------- MARKET METHODS -----------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#  
    
    def get_market(self, market: str = 'BTC-PERP'):
        return self.get('markets/' + market)
    
    def get_orderbook(self, market: str = 'BTC-PERP', depth: int = 20):
        return self.get('markets', {'market_name': market,
                                    'depth': depth})
    def get_historical_prices(self, start_time: int, end_time: int, market: str = 'BTC-PERP', resolution: int = 86400, ):    
        return self.get('markets/' + market + '/candles?resolution=' + str(resolution) + '&start_time=' +
                        str(start_time) + '&end_time=' + str(end_time))
    