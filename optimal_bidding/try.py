from nemosis import data_fetch_methods

start_time = '2018/06/01 04:00:00'
end_time = '2018/07/01 04:00:00'
table = 'DISPATCHPRICE'
raw_data_cache = 'static'

price_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, table, raw_data_cache)
